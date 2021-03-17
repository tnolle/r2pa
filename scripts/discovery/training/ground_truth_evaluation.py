from april import Dataset
from april.fs import get_event_log_files
from r2pa.discovery.evaluation import evaluate
from r2pa.discovery.process_model import ProcessModel


def evaluate_from_ground_truth(dataset):
    dataset = Dataset(dataset, use_event_attributes=True)
    ground_truth_model = ProcessModel(dataset=dataset)

    # group attribute nodes
    evaluate(dataset=dataset, model=None, ground_truth_model=ground_truth_model,
             use_cache=False, cache_generation_function=None, next_events_generation_function=None,
             next_event_threshold=0.00, id="gt", use_ground_truth_cases_input=True, group_attribute_nodes=True)

    # do not group attribute nodes
    evaluate(dataset=dataset, model=None, ground_truth_model=ground_truth_model,
             use_cache=False, cache_generation_function=None, next_events_generation_function=None,
             next_event_threshold=0.00, id="gt", use_ground_truth_cases_input=True, group_attribute_nodes=False)


def main():
    # note: order evaluation by increasing number of attributes because those complete the fastest
    datasets = sorted([e.name for e in get_event_log_files() if e.id == 1 and e.p == 0.3])

    datasets = [
        # 'paper-0.3-1',
        # 'papermanual-0.3-1',
        'p2p-0.3-1',
        'small-0.3-1',
        'medium-0.3-1',
        'wide-0.3-1',
        'huge-0.3-1',
        # 'large-0.3-1',
        # 'gigantic-0.3-1',
    ]

    print(datasets)

    for dataset in datasets:
        evaluate_from_ground_truth(dataset)


if __name__ == '__main__':
    main()
