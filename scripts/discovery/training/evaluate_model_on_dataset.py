import traceback

import numpy as np

from r2pa.discovery.evaluation import evaluate
from r2pa.discovery.discovery import ExceedSequenceLengthError, EmptyLikelihoodGraphError


def evaluate_models(configuration):
    """
        :param configuration: (dataset, model, ground truth model,
                               next event generation function with cache, next event generation function without cache,
                               cache generation function, identifier)
    """

    # do once with cache and not grouping attribute nodes to get threshold
    threshold, threshold_delta = determine_threshold(configuration)
    number_increased_threshold = 2

    # cancel
    if threshold == 0.0:
        return

    # unpack configuration
    dataset, model, ground_truth_model, \
    next_events_generation_function_with_cache, next_events_generation_function_without_cache, \
    cache_generation_function, identifier = configuration

    # for 2 or fewer attributes, grouping attribute nodes is not necessary
    if dataset.num_attributes > 2:
        # do once with cache and grouping attribute nodes with same threshold
        evaluate(dataset=dataset, model=model, ground_truth_model=ground_truth_model,
                 next_events_generation_function=next_events_generation_function_with_cache, next_event_threshold=threshold,
                 use_cache=True, cache_generation_function=cache_generation_function,
                 id=identifier, group_attribute_nodes=True)

    # increase next event threshold
    for i in range(1, number_increased_threshold+1):
        evaluate(dataset=dataset, model=model, ground_truth_model=ground_truth_model,
                 next_events_generation_function=next_events_generation_function_with_cache,
                 next_event_threshold=np.round(threshold + i * threshold_delta, 2),
                 use_cache=True, cache_generation_function=cache_generation_function,
                 id=identifier, group_attribute_nodes=False)

    # do once without cache and not grouping attribute to get speedup when using cache
    evaluate(dataset=dataset, model=model, ground_truth_model=ground_truth_model,
             next_events_generation_function=next_events_generation_function_without_cache, next_event_threshold=threshold,
             use_cache=False, cache_generation_function=cache_generation_function,
             id=identifier, group_attribute_nodes=False)


def determine_threshold(configuration):
    """
        :param configuration: (dataset, model, ground truth model, next event generation function,
                               cache generation function, identifier)
        :return: A low, possible threshold with one evaluation already finished (without grouping attribute nodes).
    """
    # starting threshold
    threshold = 0.02
    while True:
        # greater threshold does not make sense
        if threshold > 1.0:
            print(f"Model {configuration[1].file_name} reached a threshold of > 1.0. Terminate.")
            return 0.0, 0.0

        # adjust threshold delta -> 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.30, 0.35,...
        if threshold >= 0.50:
            threshold_delta = 0.10
        elif threshold >= 0.20:
            threshold_delta = 0.05
        else:
            threshold_delta = 0.02

        try:
            print(f"trying next event threshold {threshold}")
            # note: determine threshold when not grouping attribute nodes (faster)
            dataset, model, ground_truth_model, next_events_function_with_cache, _, \
            cache_generation_function, identifier = configuration
            # call with current threshold
            evaluate(dataset=dataset, model=model, ground_truth_model=ground_truth_model,
                     next_events_generation_function=next_events_function_with_cache, next_event_threshold=threshold,
                     use_cache=True, cache_generation_function=cache_generation_function,
                     id=identifier, group_attribute_nodes=False)
            # if it runs through, done, store threshold somewhere? -> is stored in evaluation result file
            return threshold, threshold_delta
        except ExceedSequenceLengthError:
            # threshold too low
            # increase threshold by fixed value, e.g. 0.01 or 0.02 and try again
            # round to two decimals due to floating point numbers
            print(traceback.format_exc())
            threshold = np.round(threshold + threshold_delta, 2)
            continue
        except EmptyLikelihoodGraphError:
            # threshold too high, cancel
            print(traceback.format_exc())
            print(f"Model {configuration} did not run through due to too high threshold. Terminate.")
            return 0.0, 0.0
        except MemoryError:
            # do not really know if threshold works, thus increase and try again
            # round to two decimals due to floating point numbers
            print(traceback.format_exc())
            threshold = np.round(threshold + threshold_delta, 2)
            continue
        except Exception:
            # every other exception, cancel and print
            print(traceback.format_exc())
            print(f"Model {configuration} did not run through. Terminate.")
            return 0.0, 0.0
