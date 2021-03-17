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
import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from april.database import get_engine, Model, EventLog
from april.dataset import Dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from r2pa.transformer.transformer import TransformerModel


def continue_fit_and_save(dataset_name, ad, model_file_name, ad_kwargs=None, fit_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    # Save start time
    start_time = arrow.now()

    # Dataset
    dataset = Dataset(dataset_name, use_event_attributes=True)

    # AD
    import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)
    ad = ad(dataset=dataset, num_encoder_layers=6, mha_heads=6, **ad_kwargs)
    # for continuing training
    ad.load(f'{MODEL_DIR}/{model_file_name}.h5')

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean()

    # train_data = np.array(dataset.hierarchic_features)
    train_data = np.array(dataset.features)

    train_targets = dataset.get_targets_one_hot_soft(fit_kwargs['smoothing_extend']) \
        if fit_kwargs['smoothing_extend'] else dataset.targets_one_hot

    # loss_weights = {'name': 1, '0cost': 0.05, 'activity': 1, 'user': 2}

    batch_size = 50
    # Iterate over the batches of a dataset.
    for epoch in range(fit_kwargs['epochs']):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step in range(int(train_data.shape[1] / batch_size)):
            x_batch_train = train_data[:, batch_size * step:batch_size * (step + 1)]
            y_batch_train = [t[batch_size * step:batch_size * (step + 1)] for t in train_targets]
            x_batch_train_ = [t for t in x_batch_train]

            with tf.GradientTape() as tape:
                reconstructed, _, _ = ad(x_batch_train_, training=True)

                loss = 0
                for key, t, r in zip(dataset.attribute_keys, y_batch_train, reconstructed):
                    # loss += loss_weights[key] * loss_fn(t, r)
                    loss += loss_fn(t, r)

                loss /= len(x_batch_train_)
                loss += sum(ad.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, ad.trainable_weights)
            optimizer.apply_gradients(zip(grads, ad.trainable_weights))

            loss_metric(loss)

            if step % 10 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))

    import os
    file_name = f'{dataset_name}_{ad.abbreviation}_{start_time.format(DATE_FORMAT)}'
    file_path = os.path.join(
            '..',
            '..',
            '.out',
            'models',
            file_name + '.h5'
        )
    ad.save_weights(file_path)

    # Save end time
    end_time = arrow.now()

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

    tf.keras.backend.clear_session()


def fit_and_save(dataset_name, ad, ad_kwargs=None, fit_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    # Save start time
    start_time = arrow.now()

    # Dataset
    dataset = Dataset(dataset_name, use_event_attributes=True)

    # AD
    import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)
    ad = ad(dataset=dataset, num_encoder_layers=6, mha_heads=6, **ad_kwargs)
    # for continuing training
    # ad.load('../../.out/models/.h5')

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean()

    # train_data = np.array(dataset.hierarchic_features)
    train_data = np.array(dataset.features)

    train_targets = dataset.get_targets_one_hot_soft(fit_kwargs['smoothing_extend']) \
        if fit_kwargs['smoothing_extend'] else dataset.targets_one_hot

    # loss_weights = {'name': 1, '0cost': 0.05, 'activity': 1, 'user': 2}

    batch_size = 50
    # Iterate over the batches of a dataset.
    for epoch in range(fit_kwargs['epochs']):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step in range(int(train_data.shape[1] / batch_size)):
            x_batch_train = train_data[:, batch_size * step:batch_size * (step + 1)]
            y_batch_train = [t[batch_size * step:batch_size * (step + 1)] for t in train_targets]
            x_batch_train_ = [t for t in x_batch_train]

            with tf.GradientTape() as tape:
                reconstructed, _, _ = ad(x_batch_train_, training=True)

                loss = 0
                for key, t, r in zip(dataset.attribute_keys, y_batch_train, reconstructed):
                    # loss += loss_weights[key] * loss_fn(t, r)
                    loss += loss_fn(t, r)

                loss /= len(x_batch_train_)
                loss += sum(ad.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, ad.trainable_weights)
            optimizer.apply_gradients(zip(grads, ad.trainable_weights))

            loss_metric(loss)

            if step % 10 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))

    import os
    file_name = f'{dataset_name}_{ad.abbreviation}_{start_time.format(DATE_FORMAT)}'
    file_path = os.path.join(
            '..',
            '..',
            '.out',
            'models',
            file_name + '.h5'
        )
    ad.save_weights(file_path)

    # Save end time
    end_time = arrow.now()

    # Cache result

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

    tf.keras.backend.clear_session()


def main():
    # datasets = sorted([e.name for e in get_event_log_files()])
    datasets = ['medium-0.3-1']

    sequential_neps = [
        dict(ad=TransformerModel,
             fit_kwargs=dict(epochs=40, batch_size=50, validation_split=0.1, smoothing_extend=0.1)),
    ]
    
    # NN ADs
    for ad in sequential_neps:
        [fit_and_save(d, **ad) for d in tqdm(datasets, desc=ad['ad'].name)]


if __name__ == '__main__':
    try:
        main()
        # notify(f'{__file__} on `{socket.gethostname()}` has finished')
    except Exception as err:
        # notify(f'{__file__} on `{socket.gethostname()}` has crashed with the following '
        #        f'message.\n\n```\n{err}\n```')
        raise err
