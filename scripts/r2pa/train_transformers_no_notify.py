# Copyright 2019 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import socket

import arrow
from sqlalchemy.orm import Session
from tqdm import tqdm

import numpy as np
from april.database import get_engine, Model
from april.dataset import Dataset
from april.fs import MODEL_DIR, DATE_FORMAT
from april.processmining import EventLog


# from scripts.notifications import notify
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from r2pa.transformer.transformer import TransformerModel


def fit_and_save(use_event_attributes, dataset_name, ad, ad_kwargs=None, fit_kwargs=None):
    import tensorflow as tf

    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    # whether to use the validation set loss for early stopping or the training set loss
    use_val_loss = True

    # Save start time
    start_time = arrow.now()

    # Dataset
    dataset = Dataset(dataset_name, use_event_attributes=use_event_attributes)

    # tf.config.experimental_run_functions_eagerly(True)
    ad = ad(dataset=dataset, num_encoder_layers=ad_kwargs['num_enc_layers'], mha_heads=ad_kwargs['num_heads'])

    # early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
    callback.set_model(ad)

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_val = tf.keras.metrics.Mean()

    data = np.array(dataset.features)

    # validation split and shuffling
    validation_split = fit_kwargs['validation_split'] if fit_kwargs['validation_split'] else 0
    train_split = int(dataset.num_cases * (1 - validation_split))

    if validation_split > 0:
        random_generator = np.random.default_rng(seed=48)
        random_generator.shuffle(data, axis=1)

    train_data = data[:, :train_split, :]
    val_data = data[:, train_split:, :]

    targets = dataset.get_targets_one_hot_soft(fit_kwargs['smoothing_extend']) \
        if fit_kwargs['smoothing_extend'] else dataset.targets_one_hot

    if validation_split > 0:
        for attribute in range(dataset.num_attributes):
            random_generator = np.random.default_rng(seed=48)
            random_generator.shuffle(targets[attribute], axis=0)

    train_targets = [attribute_targets[:train_split, :, :] for attribute_targets in targets]
    val_targets = [attribute_targets[train_split:, :, :] for attribute_targets in targets]

    callback.on_train_begin()  # for early stopping

    batch_size = 50
    # Iterate over the batches of a dataset.
    for epoch in range(fit_kwargs['epochs']):
        print('Start of epoch %d' % (epoch,))
        
        callback.on_epoch_begin(epoch=epoch)

        # Iterate over the batches of the dataset.
        for step in range(int(train_data.shape[1] / batch_size)):
            callback.on_batch_begin(batch=step)  # for early stopping
            
            x_batch_train = train_data[:, batch_size * step:batch_size * (step + 1)]
            y_batch_train = [t[batch_size * step:batch_size * (step + 1)] for t in train_targets]
            x_batch_train_ = [t for t in x_batch_train]

            with tf.GradientTape() as tape:
                # get loss on training data
                reconstructed, _, _ = ad(x_batch_train_, training=True)

                loss = 0
                for key, t, r in zip(dataset.attribute_keys, y_batch_train, reconstructed):
                    loss += loss_fn(t, r)

                loss /= len(x_batch_train_)
                loss += sum(ad.losses)  # Add KLD regularization loss

            # get validation split loss
            reconstructed_validation, _, _ = ad([t for t in val_data])

            # calculate validation loss
            if validation_split > 0:
                val_loss = 0
                for key, t, r in zip(dataset.attribute_keys, val_targets, reconstructed_validation):
                    val_loss += loss_fn(t, r)
                val_loss /= len(val_data)
                val_loss += sum(ad.losses)

                loss_metric_val(val_loss)
                loss_metric_val_result = loss_metric_val.result()

            grads = tape.gradient(loss, ad.trainable_weights)
            optimizer.apply_gradients(zip(grads, ad.trainable_weights))

            loss_metric(loss)
            loss_metric_result = loss_metric.result()

            if step % 10 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric_result))
                if validation_split > 0:
                    print('step %s: mean val loss 0 %s' % (step, loss_metric_val_result))

            callback.on_batch_end(batch=step)  # for early stopping

        # for early stopping
        callback.on_epoch_end(epoch=epoch,
                              logs={'loss': loss_metric_val_result if use_val_loss else loss_metric_result})
        print("callback", callback.wait, callback.stopped_epoch)

        if callback.stopped_epoch > 0:
            break

    callback.on_train_end()  # for early stopping

    import os
    file_name = f'{dataset_name}_{ad.abbreviation}_{start_time.format(DATE_FORMAT)}'
    file_path = os.path.join(str(MODEL_DIR / file_name) + '.h5')
    ad.save_weights(file_path)

    # Save end time
    end_time = arrow.now()

    # Cache result
    #Evaluator(file_name).cache_result()

    # Calculate training time in seconds
    training_time = (end_time - start_time).total_seconds()

    # Write to database
    engine = get_engine()
    session = Session(engine)

    session.add(Model(creation_date=end_time.datetime,
                      algorithm=ad.name,
                      training_duration=training_time,
                      file_name=file_path,  # model_file.file,
                      training_event_log_id=EventLog.get_id_by_name(dataset_name),
                      training_host=socket.gethostname(),
                      hyperparameters=str(dict(**ad_kwargs, **fit_kwargs))))
    session.commit()
    session.close()

    import tensorflow as tf
    tf.keras.backend.clear_session()


def main():
    datasets = ['papermanual-0.3-1']

    number_encoding_layers = 3
    number_heads = 3
    max_number_epochs = 50

    batch_size = 50

    datasets_models = []
    for dataset in datasets:
            datasets_models.append(
                (dataset, dict(ad=TransformerModel, ad_kwargs=dict(num_enc_layers=number_encoding_layers, num_heads=number_heads),
                 fit_kwargs=dict(epochs=max_number_epochs, batch_size=batch_size, validation_split=0.2, smoothing_extend=0.0))))

    [fit_and_save(True, d, **ad) for d, ad in tqdm(datasets_models)]


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
