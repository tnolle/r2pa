import networkx as nx

from april import Dataset
from april.fs import EVALUATION_DIR, EVENTLOG_DIR
from r2pa.discovery import discovery, drawing
from r2pa.discovery.discovery import ProcessDiscoveryResult
from r2pa.discovery.evaluation import evaluate
from r2pa.discovery.next_event_prediction.transformer import TransformerNextEventPredictor
from r2pa.transformer.transformer import TransformerModel


def get_transformer_model(dataset, model, number_layers, number_heads):
    transformer = TransformerModel(dataset, num_encoder_layers=number_layers, mha_heads=number_heads)
    from april.fs import MODEL_DIR
    file_name = model + '.h5'
    transformer.load(MODEL_DIR / file_name)
    return transformer


if __name__ == '__main__':
    dataset = Dataset('papermanual-0.3-1', use_event_attributes=True, use_case_attributes=False)
    transformer = get_transformer_model(dataset, 'papermanual-0.3-1_Transformer_20210226-172440.552275', 3, 3)

    # pr = cProfile.Profile()
    # pr.enable()

    file_name = 'graph_output'

    # discovery
    next_event_predictor = TransformerNextEventPredictor(dataset=dataset, model=transformer,
                                                         next_event_threshold=0.03, use_cache=False)

    discovery_result = discovery.discover_graph_using_next_event_predictor(dataset=dataset, group_attribute_nodes=True,
                                                                           next_event_predictor=next_event_predictor)
    discovery_result.store(EVALUATION_DIR / file_name)

    drawing.draw_and_store_likelihood_graph_with_colors(discovery_result.graph, dataset, file_name=file_name,
                                                        coder_attributes=dataset.get_encoder_decoder_for_attributes())


    # evaluation
    discovery_result = ProcessDiscoveryResult.load(EVALUATION_DIR / file_name)

    ground_truth_process_model = nx.read_gpickle(EVENTLOG_DIR / f'graph_{dataset.dataset_name}.gpickle')
    evaluate(name=file_name + '_evaluation', dataset=dataset, ground_truth_process_model=ground_truth_process_model,
             process_discovery_result=discovery_result)

    # pr.disable()
    # pr.print_stats(sort='cumulative')
