import pickle
import socket
from collections import defaultdict

import numpy as np
import networkx as nx

import arrow
from sqlalchemy.orm import Session
import traceback

import r2pa
from april.database import Model
from april.database import get_engine
from r2pa.discovery import discovery, drawing
from r2pa.discovery.automatable_procedures import AutomatableProcedures
from r2pa.discovery.discovery import ExceedSequenceLengthError, EmptyLikelihoodGraphError, ProcessDiscoveryResult
from april.fs import MODEL_DIR,  EVALUATION_DIR, EVENTLOG_DIR

import tensorflow as tf

from april import Dataset, fs
from april.alignments.binet import BINet
from april.processmining import EventLog
from r2pa.discovery.next_event_prediction.binetv1 import BINetV1NextEventPredictor
from r2pa.discovery.next_event_prediction.binetv2 import BINetV2NextEventPredictor
from r2pa.discovery.next_event_prediction.binetv3 import BINetV3NextEventPredictor
from r2pa.discovery.next_event_prediction.transformer import TransformerNextEventPredictor
from r2pa.transformer.transformer import TransformerModel


def get_present_setting(version):
    present_settings = {1: (False, False), 2: (True, False), 3: (True, True)}
    combination = {1: '00', 2: '10', 3: '11'}
    return present_settings[version], combination[version]


def train_binet(output_locations, event_log, version, parameters):
    """ Train a BINet with the given parameters. """
    dataset = Dataset(event_log, use_event_attributes=True, use_case_attributes=False)

    (present_activity, present_attribute), _ = get_present_setting(version)
    binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                  use_present_activity=present_activity, use_present_attributes=present_attribute)

    x, y = dataset.features, dataset.targets
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    binet.fit(x=x, y=y, batch_size=parameters['batch_size'], epochs=parameters['epochs'],
              validation_split=parameters['validation_split'], callbacks=[callback])

    output_name = output_locations[0]
    binet.save_weights(str(fs.MODEL_DIR / f'{output_name}.h5'))

    tf.keras.backend.clear_session()


def train_transformer(output_locations, event_log, parameters):
    """ Train a Transformer with the given parameters. """
    import tensorflow as tf

    # whether to use the validation set loss for early stopping or the training set loss
    use_val_loss = True

    # Save start time
    start_time = arrow.now()

    # Dataset
    dataset = Dataset(event_log, use_event_attributes=parameters['use_event_attributes'])

    # tf.config.experimental_run_functions_eagerly(True)
    ad = TransformerModel(dataset=dataset,
                          num_encoder_layers=parameters['number_layers'], mha_heads=parameters['number_heads'])

    # early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=parameters['early_stopping_patience'],
                                                min_delta=parameters['early_stopping_delta'])
    callback.set_model(ad)

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_val = tf.keras.metrics.Mean()

    data = np.array(dataset.features)

    # validation split and shuffling
    validation_split = parameters['validation_split'] if parameters['validation_split'] else 0
    train_split = int(dataset.num_cases * (1 - validation_split))

    if validation_split > 0:
        random_generator = np.random.default_rng(seed=48)
        random_generator.shuffle(data, axis=1)

    train_data = data[:, :train_split, :]
    val_data = data[:, train_split:, :]

    targets = dataset.get_targets_one_hot_soft(parameters['smoothing_extend']) \
        if parameters['smoothing_extend'] else dataset.targets_one_hot

    if validation_split > 0:
        for attribute in range(dataset.num_attributes):
            random_generator = np.random.default_rng(seed=48)
            random_generator.shuffle(targets[attribute], axis=0)

    train_targets = [attribute_targets[:train_split, :, :] for attribute_targets in targets]
    val_targets = [attribute_targets[train_split:, :, :] for attribute_targets in targets]

    callback.on_train_begin()  # for early stopping

    batch_size = parameters['batch_size']
    # Iterate over the batches of a dataset.
    for epoch in range(parameters['epochs']):
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
    file_name = output_locations[0]
    file_path = os.path.join(str(MODEL_DIR / file_name) + '.h5')
    # ad.save(os.path.join(str(MODEL_DIR / file_name)))
    ad.save_weights(file_path)

    # Save end time
    end_time = arrow.now()

    # Cache result
    # Evaluator(file_name).cache_result()

    # Calculate training time in seconds
    training_time = (end_time - start_time).total_seconds()

    # Write to database
    engine = get_engine()
    session = Session(engine)

    session.add(Model(creation_date=end_time.datetime,
                      algorithm=ad.name,
                      training_duration=training_time,
                      file_name=file_path,  # model_file.file,
                      training_event_log_id=EventLog.get_id_by_name(event_log),
                      training_host=socket.gethostname(),
                      hyperparameters=str(parameters)))
    session.commit()
    session.close()

    tf.keras.backend.clear_session()


def get_binet_model(dataset, model, use_present_activity, use_present_attributes):
    """ Loads the specified BINet model. """
    binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                  use_present_activity=use_present_activity, use_present_attributes=use_present_attributes)
    binet([f[:1] for f in dataset.features])
    binet.load_weights(str(MODEL_DIR / model) + ".h5")
    return binet


def get_transformer_model(dataset, model):
    model_parameters = model.split('_')[1]
    number_layers, number_heads = model_parameters.split['TR']
    transformer = TransformerModel(dataset, num_encoder_layers=number_layers, mha_heads=number_heads)
    from april.fs import MODEL_DIR
    file_name = model + '.h5'
    transformer.load(MODEL_DIR / file_name)
    return transformer, number_layers, number_heads


def determine_threshold(dataset, next_event_predictor, group_attribute_nodes):
    """
        :param configuration: (dataset, model, ground truth model, next event generation function,
                               cache generation function, identifier)
        :return: A low, possible threshold with one evaluation already finished (without grouping attribute nodes).
    """
    threshold = 0.01
    while True:
        if threshold > 1.0:
            print("Model reached a threshold of > 1.0. Terminate.")
            return 0.0, 0.0

        # adjust threshold delta the higher the threshold
        if threshold >= 0.50:
            threshold_delta = 0.05
        elif threshold >= 0.20:
            threshold_delta = 0.02
        else:
            threshold_delta = 0.01

        try:
            print(f"\nTrying next event threshold {threshold}..")
            # call with current threshold
            next_event_predictor.clear_cached_cases()
            next_event_predictor.next_event_threshold = threshold
            discovery_result = r2pa.discovery.discovery.\
                discover_graph_using_next_event_predictor(dataset=dataset, next_event_predictor=next_event_predictor,
                                                          group_attribute_nodes=group_attribute_nodes)
            # if it runs through, done
            return discovery_result, threshold
        except ExceedSequenceLengthError:
            # threshold too low
            # increase threshold by fixed value, e.g. 0.01 or 0.02 and try again
            # round to two decimals due to floating point numbers
            # print(traceback.format_exc())
            print("The length of a generated walk exceeded the maximum sequence length.")
            threshold = np.round(threshold + threshold_delta, 2)
            continue
        except EmptyLikelihoodGraphError:
            # threshold too high, cancel
            # print(traceback.format_exc())
            print(f"Model did not run through due to too high threshold. Terminate.")
            return None, 0.0
        except MemoryError:
            # do not really know if threshold works, thus increase and try again
            # round to two decimals due to floating point numbers
            # print(traceback.format_exc())
            print("Insufficient memory... increasing threshold might help mitigate this problem.")
            threshold = np.round(threshold + threshold_delta, 2)
            continue
        except Exception:
            # every other exception, cancel and print
            print(traceback.format_exc())
            print(f"Model did not run through. Terminate.")
            return None, 0.0


def graph_add_display_names(graph, coder_attributes):
    """ Add display names to a graph.
        :param graph: The graph for which the display names are to be added.
        :param coder_attributes: Decoding labels to display name. """
    # add identifier attribute
    for node in graph.nodes:
        node_attributes = graph.nodes[node]
        identifier = coder_attributes.decode(node_attributes['label'], node_attributes['attribute'])
        node_attributes['display_name'] = identifier


def discovery(output_locations, event_log, model, next_event_threshold, use_cache, group_attribute_nodes):
    dataset = Dataset(event_log, use_event_attributes=True)
    file_name = output_locations[0]

    # create next event predictor from given model
    if 'binet' in model.lower():
        use_present_activity, use_present_attributes = [bool(int(o)) for o in model.split('_')[1][-2:]]
        binet = get_binet_model(dataset, model, use_present_activity, use_present_attributes)
        if use_present_activity and use_present_attributes:
            next_event_predictor = BINetV3NextEventPredictor(dataset=dataset, model=binet, use_cache=use_cache,
                                                             next_event_threshold=next_event_threshold)
        elif use_present_activity and not use_present_attributes:
            next_event_predictor = BINetV2NextEventPredictor(dataset=dataset, model=binet, use_cache=use_cache,
                                                             next_event_threshold=next_event_threshold)
        else:
            next_event_predictor = BINetV1NextEventPredictor(dataset=dataset, model=binet, use_cache=use_cache,
                                                             next_event_threshold=next_event_threshold)
    else:
        transformer, _, _ = get_transformer_model(dataset, model)
        next_event_predictor = TransformerNextEventPredictor(dataset=dataset, model=transformer, use_cache=use_cache,
                                                             next_event_threshold=next_event_threshold)

    if next_event_threshold == -1:
        discovery_result, threshold = determine_threshold(dataset, next_event_predictor, group_attribute_nodes)
    else:
        # discover process model and store results
        discovery_result = r2pa.discovery.discovery.\
            discover_graph_using_next_event_predictor(dataset=dataset, next_event_predictor=next_event_predictor,
                                                      group_attribute_nodes=group_attribute_nodes)

    discovery_result.store(EVALUATION_DIR / file_name)
    drawing.draw_and_store_likelihood_graph_with_colors(discovery_result.graph, dataset, file_name=file_name,
                                                        coder_attributes=dataset.get_encoder_decoder_for_attributes())


def find_automatable_procedures_from_graph(output_locations, file_name, minimum_sequence_length, minimum_edge_value):
    # load graph from file
    graph = nx.read_gpickle(f"{fs.EVALUATION_DIR / file_name}.gpickle")
    # find automatable procedures
    automatable_procedures = AutomatableProcedures(graph=graph, minimum_sequence_length=minimum_sequence_length,
                                                   minimum_edge_value=minimum_edge_value)
    procedures = automatable_procedures.find()
    # convert node identifiers to display names
    decoded_procedures = [([graph.nodes[n]['display_name'] for n in procedure], likelihoods)
                          for procedure, likelihoods in procedures]

    # store decoded automatable procedures?
    return decoded_procedures


def evaluate(output_locations, event_log, file_name):
    dataset = Dataset(event_log, use_event_attributes=True)

    discovery_result = ProcessDiscoveryResult.load(EVALUATION_DIR / file_name)

    from r2pa.discovery.evaluation import evaluate as evaluate_result
    ground_truth_process_model = nx.read_gpickle(EVENTLOG_DIR / f'graph_{dataset.dataset_name}.gpickle')
    evaluate_result(name=output_locations[0], dataset=dataset,
                    ground_truth_process_model=ground_truth_process_model,
                    process_discovery_result=discovery_result)
