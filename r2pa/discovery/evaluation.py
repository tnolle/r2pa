import multiprocessing as mp
import pickle
from collections import defaultdict

import arrow
import editdistance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import r2pa.discovery.case_utils as walk_utils
from r2pa.discovery import discovery
from r2pa.discovery.coder import EncodingDecodingAttributes
from april.fs import EVALUATION_DIR, WALKS_DIR


def determine_levenshtein_distance(comparison_walks, walk_to_test):
    """ Determines the Levenshtein distance of walks_to_test to comparison_walks and sums them up.
        Assumes that walks_to_test does not contain any walks from comparison_walks (cf. minimum distance of 1).
        :param comparison_walks: The walks to compare to, e.g. ground truth walks
        :param walk_to_test: The walks to test, e.g. model walks
        :return: The sum of levenshtein distances of walks_to_test """
    minimum_distance = np.inf
    # determine most similar walk and how many adjustments are needed to convert one to the other
    for comparison_walk in comparison_walks:
        distance = editdistance.eval(walk_to_test, comparison_walk)
        if distance < minimum_distance:
            minimum_distance = distance
        # we know that 1 is the minimum distance because we only feed walks that are not in ground truth anyway
        if minimum_distance == 1:
            break
    return minimum_distance


def walk_length_weighted_levenshtein_distance(ground_truth_walks, model_without_ground_truth_walks, intersection_walks):
    """ Basically fitness/precision.
        :param ground_truth_walks:
        :param model_without_ground_truth_walks:
        :param intersection_walks:
        :return: """
    # incorporates structure of walk
    # distance of walks to walks generated by the model for walks that are not generated by the model
    # for walks also generated by the model, the distance is 0
    # now weight by the length of walks
    # TODO: disregard start and end symbols?! is always there, not really something important
    print("levenshtein distance", len(ground_truth_walks), len(model_without_ground_truth_walks),
          len(intersection_walks))

    pool = mp.Pool()
    from functools import partial
    func = partial(determine_levenshtein_distance, ground_truth_walks)

    results = pool.map_async(func, model_without_ground_truth_walks)
    model_without_ground_distance = sum(results.get())
    number_model_without_ground_distance = len(model_without_ground_truth_walks)

    average_distance_incorrect_walks = 0
    # average across walks and round
    if number_model_without_ground_distance > 0:
        average_distance_incorrect_walks = np.round(model_without_ground_distance / number_model_without_ground_distance, 4)

    # get length of all walks generated by the model,
    total_length_of_walks = 0
    total_length_of_walks += sum(len(w) for w in intersection_walks)
    total_length_of_walks += sum(len(w) for w in model_without_ground_truth_walks)

    percentage = np.round(1 - (model_without_ground_distance / total_length_of_walks), 4)

    return percentage, average_distance_incorrect_walks, model_without_ground_distance, total_length_of_walks


def compare_case_likelihoods(group_walks, remove_events=(0, 0), draw_distributions=False):
    """ Calculates the difference between likelihoods of entire walks. Averages across walks.
        :param group_walks: A dictionary with flattened walk string as key and the two (normal) walks as items
        :param draw_distributions: Whether to draw the distributions of the differences
        :return: absolute likelihood differences, normalized likelihood differences """
    print("case likelihoods")
    number_walks, total_difference, normalized_distance = len(group_walks), 0, 0
    # percentage of ground truth walk likelihood
    normalized_likelihood_differences, likelihood_differences = np.zeros(number_walks), np.zeros(number_walks)
    for index, (walk_string, walks) in enumerate(group_walks.items()):
        flattened_model_walk_likelihood = walk_utils.flatten_walk(walks[0][1, :, :])
        # remove padding from model walks
        model_walk = flattened_model_walk_likelihood[:flattened_model_walk_likelihood.index(0)]
        model_walk = model_walk[remove_events[0]:-remove_events[1]]
        model_walk_likelihood = np.prod(np.array(model_walk))
        # flatten ground truth walk and remove padding
        ground_truth_walk = walk_utils.flatten_walk(walks[1][1])
        ground_truth_walk = ground_truth_walk[:ground_truth_walk.index(0)]
        ground_truth_walk = ground_truth_walk[remove_events[0]:-remove_events[1]]
        ground_truth_walk_likelihood = np.prod(np.array(ground_truth_walk))
        # calculate difference
        likelihood_difference = np.abs(np.subtract(ground_truth_walk_likelihood, model_walk_likelihood))
        total_difference += likelihood_difference
        likelihood_differences[index] = likelihood_difference
        # calculate normalized deviation w.r.t ground truth likelihood
        normalized_likelihood = likelihood_difference / ground_truth_walk_likelihood
        normalized_distance += normalized_likelihood
        normalized_likelihood_differences[index] = normalized_likelihood

    # average likelihoods across walks
    likelihoods = total_difference / number_walks if number_walks > 0 else 0
    normalized_likelihoods = normalized_distance / number_walks if number_walks > 0 else 0

    if draw_distributions:
        plt.figure()
        plt.title("Normalized Walk Likelihood Differences")
        plt.hist(normalized_likelihood_differences)
        plt.show()
        plt.close()

        plt.figure()
        plt.title("Absolute Walk Likelihood Differences")
        plt.hist(likelihood_differences)
        plt.show()
        plt.close()

    return likelihoods, normalized_likelihoods


def compare_individual_likelihoods(group_walks, remove_events=(0, 0), draw_distributions=False):
    """ Calculates the differences between individual likelihoods of two walks of same length.
        Averages across walks and items, i.e. result is MSE per likelihood.
        :param group_walks: A dictionary with flattened walk string as key and the two (normal) walks as items.
        :param draw_distributions: Whether to draw the distributions of the differences.
        :return: absolute likelihood difference, mean squared likelihood difference """
    print("individual case likelihoods")
    number_walks, sum_likelihood_differences, likelihood_differences, mean_squared_error = len(group_walks), 0, [], []
    for walk_string, walks in group_walks.items():
        model_walk_likelihoods = walk_utils.flatten_walk(walks[0][1, :, :])
        # remove padding from model walks
        model_walk_likelihoods = model_walk_likelihoods[:model_walk_likelihoods.index(0)]
        model_walk_likelihoods = model_walk_likelihoods[remove_events[0]:-remove_events[1]]
        # remove padding and flatten ground truth walk
        ground_truth_walk_likelihoods = walk_utils.flatten_walk(walks[1][1, :, :])
        ground_truth_walk_likelihoods = ground_truth_walk_likelihoods[:ground_truth_walk_likelihoods.index(0)]
        ground_truth_walk_likelihoods = ground_truth_walk_likelihoods[remove_events[0]:-remove_events[1]]
        # calculate differences
        likelihood_difference = np.abs(np.subtract(model_walk_likelihoods, ground_truth_walk_likelihoods))
        likelihood_differences.extend(likelihood_difference.tolist())
        mean_squared_error.extend(np.square(likelihood_difference))
        # sum and average across walk length
        sum_likelihood_difference = np.sum(likelihood_difference)
        sum_likelihood_differences += sum_likelihood_difference / len(model_walk_likelihoods)

    # average across number of walks
    absolute_likelihoods_differences = sum_likelihood_differences / number_walks if number_walks > 0 else 0
    mse = np.sum(mean_squared_error) / len(mean_squared_error)

    if draw_distributions:
        plt.figure()
        plt.title("Absolute Likelihood Differences (All Walks)")
        plt.hist(likelihood_differences)
        plt.show()
        plt.close()

        # this is for all walks, implying being accurate on frequent events is better than being good on infrequent ones
        plt.figure()
        plt.title("MSE Likelihoods (All Walks)")
        plt.hist(mean_squared_error)
        plt.show()
        plt.close()

    return absolute_likelihoods_differences, mse


# TODO:
def evaluate_from_cases():
    pass


def get_process_model_cases(dataset, process_model):
    """ Get the cases of a ground truth process model, from cache or generate.
        :param dataset: The dataset the process model is underlying to.
        :param process_model: A networkx directed acyclic likelihood graph. """
    # load ground truth cases for dataset if exists, otherwise generate and store
    file_name = dataset.dataset_name + ".pickle"
    import os.path
    if os.path.isfile(WALKS_DIR / file_name):
        print("load process model cases")
        with open(WALKS_DIR / file_name, 'rb') as fp:
            cases = pickle.load(fp)
    else:
        print("generate and store process model cases")

        encoding_mapping = dict()
        for node in process_model.nodes:
            node_name = process_model.nodes[node]['value']
            encoding_mapping[node] = node_name
        encoder_decoder_attributes = EncodingDecodingAttributes(decoder=encoding_mapping, coding_per_attribute=False,
                                                                create_encoder=True)

        cases, _, _, _ = discovery.get_cases_from_graph(dataset=dataset, graph=process_model,
                                                        encoder_decoder_attributes=encoder_decoder_attributes)
        with open(WALKS_DIR / file_name, 'wb') as fp:
            pickle.dump(cases, fp)
    return cases


def evaluate(name, dataset, ground_truth_process_model, process_discovery_result):
    evaluation_start = arrow.now()
    evaluation_results = pd.DataFrame()

    number_attributes = dataset.num_attributes

    # ground truth graph
    coder_ground_truth = EncodingDecodingAttributes.from_ground_truth_graph(ground_truth_process_model,
                                                                            dataset.attribute_keys.tolist(),
                                                                            node_attribute='value')

    # those are the ones used by the model as well
    coder_dataset = dataset.get_encoder_decoder_for_attributes()

    # may only use decoder in cases such as this!
    coder_truth_to_dataset = coder_ground_truth.concatenate_per_attribute(coder_dataset)

    ground_truth_cases = get_process_model_cases(dataset=dataset, process_model=ground_truth_process_model)
    # transform to labels and disregard likelihoods
    transformed_ground_truth_cases = \
        set(walk_utils.transform_event_walks(ground_truth_cases, padded_walks=True,
                                             encoder_decoder_attributes=coder_truth_to_dataset,
                                             with_start_symbol=(True, number_attributes),
                                             remove_events=(number_attributes, 1), output_format=tuple))

    # get and transform discovered cases
    discovered_cases, coder_model, _, _ = discovery.get_cases_from_graph(dataset=dataset, graph=process_discovery_result.graph)
    transformed_discovered_cases = set(walk_utils.transform_event_walks(discovered_cases, coder_model,
                                                                        with_start_symbol=(True, number_attributes),
                                                                        remove_events=(number_attributes, 1)))

    intersect = transformed_discovered_cases.intersection(transformed_ground_truth_cases)
    model_without_ground = transformed_discovered_cases.difference(transformed_ground_truth_cases)
    ground_without_model = transformed_ground_truth_cases.difference(transformed_discovered_cases)

    number_uncached_cases = 0
    number_uncached_cases_from_ground_truth = 0
    if process_discovery_result.uncached_cases is not None:
        number_uncached_cases = len(process_discovery_result.uncached_cases)
        number_uncached_cases_from_ground_truth = number_uncached_cases - len(model_without_ground)

    # keep track, first entry must add a row
    evaluation_results = evaluation_results.append({'number of model cases': len(transformed_discovered_cases)},
                                                   ignore_index=True)
    evaluation_results['shared attribute cases'] = len(intersect)
    evaluation_results['number of model walks not in ground truth cases'] = len(model_without_ground)
    evaluation_results['number of ground truth walks not in model cases'] = len(ground_without_model)
    evaluation_results['number of nodes ground truth graph'] = len(ground_truth_process_model.nodes)
    evaluation_results['number of nodes model graph'] = len(process_discovery_result.graph.nodes)
    evaluation_results['number of uncached cases'] = number_uncached_cases
    evaluation_results['number of uncached cases only ground truth cases'] = number_uncached_cases_from_ground_truth
    evaluation_results['percentage of uncached walks only ground truth cases'] = \
        np.round(number_uncached_cases_from_ground_truth / len(ground_truth_cases), 4)

    metrics_start = arrow.now()

    print("precision")
    precision_percentage, precision_average_distance, _, _ = walk_length_weighted_levenshtein_distance(
        ground_truth_walks=transformed_ground_truth_cases, model_without_ground_truth_walks=model_without_ground,
        intersection_walks=intersect)

    evaluation_results['precision'] = precision_percentage
    evaluation_results['precision average distance'] = precision_average_distance

    print("fitness")
    fitness_percentage, fitness_average_distance, _, _ = walk_length_weighted_levenshtein_distance(
        ground_truth_walks=transformed_discovered_cases, model_without_ground_truth_walks=ground_without_model,
        intersection_walks=intersect)

    evaluation_results['fitness'] = fitness_percentage
    evaluation_results['fitness average distance'] = fitness_average_distance

    F1 = 2 * (precision_percentage * fitness_percentage) / (precision_percentage + fitness_percentage)
    evaluation_results['f1 measure'] = np.round(F1, 4)

    # build dict that groups walks from ground truth and model
    group_cases = defaultdict(list)
    for case in discovered_cases:
        transformed_case = walk_utils.transform_event_walk(case[0, :, :], padded_walks=True,
                                                           with_start_symbol=(True, number_attributes),
                                                           encoder_decoder_attributes=coder_model,
                                                           remove_events=(number_attributes, 1))
        # can only compare the likelihood traces that are generated by both
        if transformed_case in intersect:
            group_cases[transformed_case].append(case)
    for case in ground_truth_cases:
        transformed_case = walk_utils.transform_event_walk(case[0], padded_walks=True,
                                                           encoder_decoder_attributes=coder_truth_to_dataset,
                                                           with_start_symbol=(True, number_attributes),
                                                           remove_events=(number_attributes, 1))
        # can only compare the likelihood traces that are generated by both
        if transformed_case in intersect:
            group_cases[transformed_case].append(case)

    # evaluate likelihoods
    case_likelihoods, normalized_case_likelihoods = compare_case_likelihoods(group_cases, remove_events=(1, 1))
    absolute_likelihoods, mse_likelihoods = compare_individual_likelihoods(group_cases, remove_events=(1, 1))

    evaluation_results['likelihood difference per case'] = case_likelihoods
    evaluation_results['normalized likelihood difference per case'] = normalized_case_likelihoods
    evaluation_results['absolute likelihood difference per case and attribute'] = absolute_likelihoods
    evaluation_results['mean squared error likelihoods'] = mse_likelihoods

    metrics_time = (arrow.now() - metrics_start).total_seconds()
    evaluation_results['calculating metrics time (seconds)'] = metrics_time

    evaluation_time = (arrow.now() - evaluation_start).total_seconds()
    evaluation_results['evaluation time (seconds)'] = evaluation_time

    evaluation_results.to_csv(EVALUATION_DIR / f'{name}.csv')

    print('evaluation completed and results stored')
