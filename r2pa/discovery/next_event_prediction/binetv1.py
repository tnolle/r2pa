import numpy as np
import itertools

import r2pa.discovery.case_utils as walk_utils
from r2pa.discovery.next_event_prediction.next_event_prediction import NextEventPredictor


class BINetV1NextEventPredictor(NextEventPredictor):

    def __init__(self, dataset, model, next_event_threshold, use_cache):
        super().__init__(dataset=dataset, model=model, next_event_threshold=next_event_threshold,
                         use_cache=use_cache, name='BINetV1')

    def next_event(self, input_sequence, input_sequence_length):
        """ Given a case, return a set of possible next events. """
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        input_case = list(input_sequence[0].astype(int))
        input_case = [x.reshape((1, sequence_length)) for x in input_case]

        prediction = self.model.call(input_case)

        # get prediction for next events
        next_events = []
        for attr in range(number_attributes):
            next_event_predictions = np.array(prediction[attr][:, input_sequence_length-1, :])\
                .reshape((attribute_dimensions[attr]))
            #next_events.append(self.event_from_prediction_by_coverage(next_event_predictions))

            next_events.append(self.event_from_prediction_by_threshold(next_event_predictions))

        next_events_attributes, next_events_likelihoods = zip(*next_events)
        event_combinations = [*itertools.product(*next_events_attributes)]
        likelihood_combinations = [*itertools.product(*next_events_likelihoods)]

        number_next_events = len(event_combinations)
        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            next_event_combinations[0, i, :] = event_combinations[i]
            next_event_combinations[1, i, :] = likelihood_combinations[i]

        return next_event_combinations

    def next_event_with_cache(self, input_sequence, input_sequence_length):
        """ Given a case, return a set of possible next events. """
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        # use cache if possible, otherwise predict using model
        input_key = walk_utils.transform_walk_to_dict_key(input_sequence[0], padded_walks=True)
        if input_key in self.cache:
            prediction = self.cache[input_key]
            cache_prediction = True
        else:
            self.uncached_cases.append(input_sequence[0])
            input_case = [*input_sequence[0].astype(int)]
            input_case = [x.reshape((1, sequence_length)) for x in input_case]
            prediction = self.model.call(input_case)
            cache_prediction = False

        # get prediction for next events
        next_events = []
        cache_entry = []
        for attr in range(number_attributes):
            # different prediction formats
            if cache_prediction:
                next_event_predictions = np.array(prediction[attr]).reshape((attribute_dimensions[attr]))
            else:
                next_event_predictions = np.array(prediction[attr][:, input_sequence_length-1, :]) \
                    .reshape((attribute_dimensions[attr]))
                cache_entry.append(self.event_from_prediction_by_threshold(next_event_predictions))
            next_events.append(self.event_from_prediction_by_threshold(next_event_predictions))

        if not cache_prediction:
            self.add_to_cache(input_key, cache_entry)

        next_events_attributes, next_events_likelihoods = zip(*next_events)
        event_combinations = [*itertools.product(*next_events_attributes)]
        likelihood_combinations = [*itertools.product(*next_events_likelihoods)]

        number_next_events = len(event_combinations)
        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            next_event_combinations[0, i, :] = event_combinations[i]
            next_event_combinations[1, i, :] = likelihood_combinations[i]

        return next_event_combinations

    def create_cache(self, perfect_start_symbol=False):
        """ Creates a cache for model predictions, i.e. mapping a sequence to the prediction likelihoods of the model.
            :param dataset: The dataset for which to create the cache.
            :param model: The model to use. Requires detect method, e.g. transformer.
            :return: A dictionary mapping an interleaved string walk to the prediction. """
        predictions = self.model.predict(self.dataset.features)
        features = self.dataset.flat_features
        cache = dict()
        for case in range(self.dataset.num_cases):
            case_features = np.transpose(features[case])
            # can stop after case length, do not need to continue caching for padding
            for index in range(self.dataset.case_lens[case]):
                index_features, index_prediction = [], []
                for attribute in range(self.dataset.num_attributes):
                    index_features.append(case_features[attribute][:index + 1])
                    prediction_index = index + 2 if perfect_start_symbol else index + 1
                    index_prediction.append(predictions[attribute][case, prediction_index - 1:prediction_index, :])
                index_key = walk_utils.transform_walk_to_dict_key(index_features, padded_walks=False)
                cache[index_key] = index_prediction
        return cache
