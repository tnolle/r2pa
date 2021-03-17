from april import Dataset
from april.alignments.binet import BINet
from r2pa.discovery import discovery
from r2pa.discovery.custom_graphs import papermanual
from r2pa.discovery.evaluation import evaluate as evaluate_from_walks


import numpy as np

from april.fs import MODEL_DIR
from r2pa.discovery.next_event_prediction.next_event_prediction import NextEventPredictor
from r2pa.discovery.process_model import ProcessModel


def get_binet_model(dataset, model, use_present_activity, use_present_attributes):
    binet = BINet(dataset, use_event_attributes=True, use_present_activity=use_present_activity,
                  use_present_attributes=use_present_attributes)

    input_sequence = np.ones((2, dataset.num_attributes, dataset.max_len), dtype=object)
    input_case = list(input_sequence[0].astype(int))
    input_case = [x.reshape((1, dataset.max_len)) for x in input_case]
    binet.call(input_case)

    binet.load_weights(str(MODEL_DIR / f"{model}.h5"))

    return binet


def get_transformer_model(dataset, model, number_layers, number_heads):
    transformer = TransformerModel(dataset, num_encoder_layers=number_layers, mha_heads=number_heads)
    transformer.load(str(MODEL_DIR / f"{model}.h5"))
    return transformer


def get_paper_manual_configs():
    evaluation_configs = []

    dataset = Dataset('papermanual-0.3-2', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    """evaluation_configs.append((dataset, get_transformer_model(dataset, 'papermanual-0.3-2_TR_20200824-124131.140523', 6, 6),
                               ground_truth_model, NextEventsGenerator.from_transformer, False,
                               graph_generation.create_detect_cache,
                               "TR_6_6", 0.3, True))

    # binetv1
    #evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-1_BINet_20200724-125845.284662', False, False),
    #                           ground_truth_model, NextEventsGenerator.from_keras_model, False,
    #                           graph_generation
    #                           .create_detect_cache,
    #                           "normal", 0.1, True))

    #evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-1_BINet_20200724-125845.284662', False, False),
    #                           ground_truth_model, NextEventsGenerator.from_keras_model_with_cache, True,
    #                           graph_generation
    #                           .create_detect_cache,
    #                           "normal_cache", 0.1, True))"""

    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_binetv2_20200826-122715.144772', True, False),
                               ground_truth_model, NextEventPredictor.from_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_activity_conditioning,
                               "present_activity_cache", 0.2, True))

    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_binetv2_20200826-122715.144772', True, False),
                               ground_truth_model, NextEventPredictor.from_conditioned_keras_model_with_cache, True,
                               discovery
                               .create_detect_cache_for_activity_conditioning,
                               "present_activity_cache", 0.2, True))
    """
    # transformer
    evaluation_configs.append((dataset, get_transformer_model(dataset, 'paper_manual-0.3-1_TR_20200601-091710.712097', 5, 5),
                               ground_truth_model, NextEventsGenerator.from_transformer_with_cache, True,
                               graph_generation
                               .create_detect_cache,
                               "TR_5_5_30_cache", 0.05, True))

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'paper_manual-0.3-1_TR_20200601-091710.712097', 5, 5),
                               ground_truth_model, NextEventsGenerator.from_transformer, False,
                               graph_generation
                               .create_detect_cache,
                               "TR_5_5_30", 0.05, True))

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'papermanual-0.3-1_TR_20200722-121411.316643', 5, 20),
                               ground_truth_model, NextEventsGenerator.from_transformer_with_cache, True,
                               graph_generation
                               .create_detect_cache,
                               "TR_5_20_20", 0.05, True))

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'papermanual-0.3-1_TR_20200722-121411.316643', 5, 20),
                               ground_truth_model, NextEventsGenerator.from_transformer, False,
                               graph_generation
                               .create_detect_cache,
                               "TR_5_20_20", 0.05, True))"""

    return evaluation_configs


def get_paper_configs():
    evaluation_configs = []

    dataset = Dataset('paper-0.3-2', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    """evaluation_configs.append((dataset, get_binet_model(dataset, 'paper-0.3-1_BINet00_20200815-021711.494340', False, False),
                               ground_truth_model,
                               NextEventsGenerator.from_keras_model_with_cache, True,
                               graph_generation.create_detect_cache, "v1", 0.1, True))"""

    evaluation_configs.append((dataset, get_binet_model(dataset, 'paper-0.3-2_BINet10_20200815-021852.328530', True, False),
                               ground_truth_model,
                               NextEventPredictor.from_conditioned_keras_model_with_cache, True,
                               discovery.create_detect_cache_for_attribute_conditioning, "v2", 0.2, True))

    """evaluation_configs.append((dataset, get_transformer_model(dataset, 'paper-0.3-1_TR_20200811-215851.159633', 6, 6),
                               ground_truth_model, NextEventsGenerator.from_transformer_with_cache, True,
                               graph_generation.create_detect_cache,
                               "TR_6_6", 0.1, True))"""
    """
    evaluation_configs.append((dataset, get_transformer_model(dataset, 'paper-0.3-1_TR_20200622-104836.630354', 8, 6),
                               ground_truth_model, NextEventsGenerator.from_transformer_with_cache, True,
                              graph_generation.create_detect_cache,
                               "TR_8_6_80", 0.15, True))

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'paper-0.3-1_TR_20200722-125749.800490', 5, 20),
                               ground_truth_model, NextEventsGenerator.from_transformer_with_cache, True,
                               graph_generation.create_detect_cache,
                               "TR_5_20_40", 0.15, True))"""

    return evaluation_configs


def get_small_configs():
    evaluation_configs = []

    dataset = Dataset('small-0.3-1', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    """evaluation_configs.append((dataset,
                               get_binet_model(dataset, 'small-0.3-1_BINet_20200802-094141.071599', False, False),
                               ground_truth_model,
                               NextEventsGenerator.from_keras_model_with_cache, True,
                               graph_generation.create_detect_cache, "v1", 0.2, True))

    evaluation_configs.append(
        (dataset, get_binet_model(dataset, 'small-0.3-1_BINet_20200802-104014.799522', True, False), ground_truth_model,
         NextEventsGenerator.from_attribute_conditioned_keras_model_with_cache, True,
         graph_generation.create_detect_cache_for_conditioning, "v2", 0.2, True))"""

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'small-0.3-1_TR_20200802-093918.933598', 8, 20),
                               ground_truth_model, NextEventPredictor.from_transformer_with_cache, True,
                               discovery.create_detect_cache,
                               "TR_8_20_40", 0.1, True))

    return evaluation_configs


def get_p2p_configs():
    evaluation_configs = []

    dataset = Dataset('p2p-0.3-1', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    """evaluation_configs.append((dataset,
                               get_binet_model(dataset, 'p2p-0.3-1_BINet_20200802-114021.062858', False, False),
                               ground_truth_model,
                               NextEventsGenerator.from_keras_model_with_cache, True,
                               graph_generation.create_detect_cache, "v1", 0.15))

    evaluation_configs.append(
        (dataset, get_binet_model(dataset, 'p2p-0.3-1_BINet_20200802-123352.009528', True, False), ground_truth_model,
         NextEventsGenerator.from_attribute_conditioned_keras_model_with_cache, True,
         graph_generation.create_detect_cache_for_conditioning, "v2", 0.15))"""

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'p2p-0.3-1_TR_20200802-151612.916686', 8, 20),
                               ground_truth_model, NextEventPredictor.from_transformer_with_cache, True,
                               discovery.create_detect_cache,
                               "TR_8_20_40", 0.1, True))

    return evaluation_configs


def get_medium_configs():
    evaluation_configs = []

    dataset = Dataset('medium-0.3-1', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    """evaluation_configs.append((dataset,
                               get_binet_model(dataset, 'medium-0.3-1_BINet_20200802-131925.274528', False, False),
                               ground_truth_model,
                               NextEventsGenerator.from_keras_model_with_cache, True,
                               graph_generation.create_detect_cache, "v1", 0.2))

    evaluation_configs.append(
        (dataset, get_binet_model(dataset, 'medium-0.3-1_BINet_20200802-134954.329528', True, False), ground_truth_model,
         NextEventsGenerator.from_attribute_conditioned_keras_model_with_cache, True,
         graph_generation.create_detect_cache_for_conditioning, "v2", 0.2))"""

    evaluation_configs.append((dataset, get_transformer_model(dataset, 'medium-0.3-1_TR_20200802-183117.846150', 8, 20),
                               ground_truth_model, NextEventPredictor.from_transformer_with_cache, True,
                               discovery.create_detect_cache,
                               "TR_8_20_40", 0.07, True))

    return evaluation_configs


def get_paper_two_attributes_config():
    evaluation_configs = []

    dataset = Dataset('papermanual-0.3-2', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    # with cache
    # v3
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-122847.768548', True, True),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model_with_cache, True,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v3", 0.2, True))

    # v2
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-150740.756240', True, False),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model_with_cache, True,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v2", 0.2, True))

    # v1
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-154907.475713', False, False),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model_with_cache, True,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v1", 0.2, True))

    #  (8, 20, 40 epochs)
    evaluation_configs.append((dataset, get_transformer_model(dataset, 'papermanual-0.3-2_TR_20200728-122541.754679', 8, 20),
                               ground_truth_model, NextEventPredictor.from_transformer_with_cache, True,
                               discovery
                               .create_detect_cache,
                               "TR_8_20_40_cache", 0.2, True))

    # no cache
    #  (8, 20, 40 epochs)
    evaluation_configs.append((dataset, get_transformer_model(dataset, 'papermanual-0.3-2_TR_20200728-122541.754679', 8, 20),
                               ground_truth_model, NextEventPredictor.from_transformer, False,
                               discovery
                               .create_detect_cache,
                               "TR_8_20_40_cache", 0.2, True))

    # v3
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-122847.768548', True, True),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v3", 0.2, True))

    # v2
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-150740.756240', True, False),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v2", 0.2, True))

    # v1
    evaluation_configs.append((dataset, get_binet_model(dataset, 'papermanual-0.3-2_BINet_20200730-154907.475713', False, False),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v1", 0.2, True))

    return evaluation_configs


def get_mini_model_config():
    evaluation_configs = []

    dataset = Dataset('mini-0.3-3', use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    # v1 mini-0.3-3_BINet_20200731-081744.912284
    evaluation_configs.append((dataset, get_binet_model(dataset, 'mini-0.3-3_BINet_20200731-081744.912284', False, False),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v1", 0.2, True))

    # v3
    evaluation_configs.append((dataset, get_binet_model(dataset, 'mini-0.3-3_BINet_20200731-082144.314410', True, True),
                               ground_truth_model, NextEventPredictor.from_attribute_conditioned_keras_model, False,
                               discovery
                               .create_detect_cache_for_attribute_conditioning,
                               "v3", 0.2, True))

    return evaluation_configs


def ground_truth_single_attribute():
    dataset = Dataset('papermanual-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset, graph=papermanual.get_directed_graph())
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('paper-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('small-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    # TODO: problem is that Activity C has 0.0 probability, therefore it is not present in the eventlog -> not in coder dataset
    """dataset = Dataset('wide-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_walks_input=True, next_event_threshold=0.05)"""

    dataset = Dataset('medium-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('p2p-0.3-1', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)


def ground_truth_two_attributes():
    dataset = Dataset('papermanual-0.3-2', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('paper-0.3-2', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('small-0.3-2', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('medium-0.3-2', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)

    dataset = Dataset('p2p-0.3-2', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)


def ground_truth_three_attributes():
    dataset = Dataset('papermanual-0.3-3', use_event_attributes=True)
    # TODO: code assumes chronological order of attributes!
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)


def ground_truth_mini():
    dataset = Dataset('mini-0.3-3', use_event_attributes=True)
    gtm = ProcessModel(dataset=dataset)
    evaluate_from_walks(dataset=dataset, model=None, use_cache=False, cache_generation_function=None,
                        ground_truth_model=gtm,
                        next_events_generation_function=None, id="from_gt",
                        use_ground_truth_cases_input=True, next_event_threshold=0.05,
                        group_attribute_nodes=True)


def ground_truth_evaluation():
    #ground_truth_single_attribute()
    ground_truth_three_attributes()
    #ground_truth_two_attributes()
    #ground_truth_mini()


def model_evaluation():
    evaluation_configs = []

    evaluation_configs.extend(get_paper_manual_configs())
    #evaluation_configs.extend(get_paper_configs())

    #evaluation_configs.extend(get_paper_two_attributes_config())
    #evaluation_configs.extend(get_mini_model_config())

    #evaluation_configs.extend(get_small_configs())
    #evaluation_configs.extend(get_p2p_configs())
    #evaluation_configs.extend(get_medium_configs())

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    for dataset, model, ground_truth_model, next_events_generation_function, use_cache, cache_generation_function, id, threshold, group_attribute_nodes in evaluation_configs:
        # try:
        evaluate_from_walks(dataset=dataset, model=model, use_cache=use_cache, ground_truth_model=ground_truth_model,
                            cache_generation_function=cache_generation_function, next_event_threshold=threshold,
                            next_events_generation_function=next_events_generation_function, id=id,
                            group_attribute_nodes=group_attribute_nodes)
    # except:
    #   pass
    # pr.disable()
    # pr.print_stats(sort='cumulative')


if __name__ == '__main__':
    #ground_truth_evaluation()
    model_evaluation()
