from april.alignments.binet import BINet
from april.dataset import Dataset
from april.fs import MODEL_DIR
from r2pa.discovery import discovery, drawing
from r2pa.discovery.next_event_prediction.next_event_prediction import NextEventPredictor

from r2pa.discovery.process_model import ProcessModel
from r2pa.transformer.transformer import TransformerModel


def get_binet_model(dataset, model, use_present_activity, use_present_attributes):
    """ Loads the specified BINet model. """
    binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                  use_present_activity=use_present_activity, use_present_attributes=use_present_attributes)
    binet([f[:1] for f in dataset.features])
    binet.load_weights(str(MODEL_DIR / model))
    return binet


def get_binet_configuration(dataset, model):
    ground_truth_model = ProcessModel(dataset=dataset)
    use_present_activity, use_present_attributes = [bool(int(o)) for o in model.file.split('_')[1][-2:]]
    model = get_binet_model(dataset, model.file, use_present_activity, use_present_attributes)
    if use_present_activity and use_present_attributes:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_attribute_conditioned_keras_model_with_cache,
                NextEventPredictor.from_attribute_conditioned_keras_model,
                discovery.create_detect_cache_for_attribute_conditioning, "BINetV3")
    elif use_present_activity and not use_present_attributes:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_conditioned_keras_model_with_cache,
                NextEventPredictor.from_conditioned_keras_model,
                discovery.create_detect_cache_for_activity_conditioning, "BINetV2")
    else:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_keras_model_with_cache,
                NextEventPredictor.from_keras_model,
                discovery.create_detect_cache, "BINetV1")


def get_transformer_model(dataset, model, number_layers, number_heads):
    transformer = TransformerModel(dataset, num_encoder_layers=number_layers, mha_heads=number_heads)
    from april import MODEL_DIR
    file_name = model + '.h5'
    transformer.load(MODEL_DIR / file_name)
    return transformer


if __name__ == '__main__':
    """dataset = Dataset('log1', use_event_attributes=True)
    transformer = get_transformer_model(dataset, 'log1_TR_20201116-092807.421127', 6, 6)"""

    dataset = Dataset('log2', use_event_attributes=True)

    # log2_binet11_20201201-172441.185216, with custom loss
    # log2_BINet11_20201121-134104.742051, without custom loss
    binet = get_binet_model(dataset, 'log2_binet11_20201201-172441.185216.h5', True, True)
    graph = discovery.create_directed_graph(
        dataset=dataset, model=binet, use_cache=False,
        cache_generation_function=discovery.create_detect_cache_for_attribute_conditioning,
        next_events_generation_function=NextEventPredictor.from_attribute_conditioned_keras_model,
        group_attribute_nodes=False, next_event_threshold=0.02)

    # transformer = get_transformer_model(dataset, 'log2_TR_20201116-142239.642218', 6, 6)
    # transformer = get_transformer_model(dataset, 'log2_TR_20201121-092915.665589', 6, 6)

    #graph = graph_generation.create_directed_graph(dataset=dataset, model=transformer, use_cache=False,
    #                                               cache_generation_function = graph_generation.create_detect_cache,
    #                                               next_events_generation_function = NextEventsGenerator.from_transformer,
    #                                               next_event_threshold = 0.2)

    file_name = "r2pa"

    # store results
    drawing.draw_and_store_likelihood_graph_with_colors(graph, dataset, file_name=file_name,
                                                        coder_attributes=dataset.get_encoder_decoder_for_attributes())
