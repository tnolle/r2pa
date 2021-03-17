import numpy as np

import r2pa.discovery.case_utils as walk_utils
from r2pa.discovery.next_event_prediction.next_event_prediction import NextEventPredictor


class BINetV3NextEventPredictor(NextEventPredictor):

    def __init__(self, dataset, model, next_event_threshold, use_cache):
        super().__init__(dataset=dataset, model=model, next_event_threshold=next_event_threshold,
                         use_cache=use_cache, name='BINetV3')

    # intended for present attributes, only sequential order
    def next_event(self, input_sequence, input_sequence_length):
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        # determine successors
        queue = [input_sequence]
        # need to return a continuation for every attribute, i.e. we need to
        for attribute in range(number_attributes):
            new_queue = []
            for sequence in queue:
                input_case = list(sequence[0].astype(int))
                input_case = [x.reshape((1, sequence_length)) for x in input_case]
                # get prediction for this sequence
                prediction = self.model.call(input_case)
                # only look at the prediction for this attribute, not for all
                next_event_predictions = np.array(prediction[attribute][:, input_sequence_length-1, :]) \
                    .reshape((attribute_dimensions[attribute]))
                next_events = self.event_from_prediction_by_threshold(next_event_predictions)
                next_events = zip(next_events[0], next_events[1])
                # generate new sequence for each possible next event
                for next_event in next_events:
                    # skip if prediction for next event is padding
                    if next_event[0] == 0:
                        continue
                    new_sequence = sequence.copy()
                    new_sequence[0][attribute][input_sequence_length] = next_event[0]
                    new_sequence[1][attribute][input_sequence_length-1] = next_event[1]  # always have one likelihood fewer
                    new_queue.append(new_sequence)
            queue = new_queue

        number_next_events = len(queue)

        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            sequence = queue[i]
            events = sequence[0][:, input_sequence_length]
            likelihoods = sequence[1][:, input_sequence_length-1]
            next_event_combinations[0, i, :] = events
            next_event_combinations[1, i, :] = likelihoods

        return next_event_combinations

    # intended for present attributes, only sequential order
    def next_event_with_cache(self, input_sequence, input_sequence_length):
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        # determine successors
        queue = [input_sequence]
        # need to return a continuation for every attribute, i.e. we need to
        for attribute in range(number_attributes):
            new_queue = []
            for sequence in queue:
                # use cache if possible, otherwise predict using model
                input_key = walk_utils.transform_walk_to_dict_key(sequence[0], padded_walks=True)
                if input_key in self.cache:
                    prediction = self.cache[input_key]
                    cache_prediction = True
                else:
                    self.uncached_cases.append(sequence[0])
                    input_case = [*sequence[0].astype(int)]
                    input_case = [x.reshape((1, sequence_length)) for x in input_case]
                    prediction = self.model.call(input_case)
                    cache_prediction = False
                # only look at the prediction for this attribute, not for all
                if cache_prediction:
                    next_event_predictions = prediction.reshape((attribute_dimensions[attribute]))
                else:
                    next_event_predictions = np.array(prediction[attribute][:, input_sequence_length-1, :]) \
                        .reshape((attribute_dimensions[attribute]))
                    # add to cache
                    self.add_to_cache(input_key, np.array(prediction[attribute][:, input_sequence_length-1, :]))
                next_events = self.event_from_prediction_by_threshold(next_event_predictions)
                next_events = zip(next_events[0], next_events[1])
                # generate new sequence for each possible next event
                for next_event in next_events:
                    # skip if prediction for next event is padding
                    if next_event[0] == 0:
                        continue
                    new_sequence = sequence.copy()
                    new_sequence[0][attribute][input_sequence_length] = next_event[0]
                    new_sequence[1][attribute][input_sequence_length - 1] = next_event[
                        1]  # always have one likelihood fewer
                    new_queue.append(new_sequence)
            queue = new_queue

        number_next_events = len(queue)

        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            sequence = queue[i]
            events = sequence[0][:, input_sequence_length]
            likelihoods = sequence[1][:, input_sequence_length - 1]
            next_event_combinations[0, i, :] = events
            next_event_combinations[1, i, :] = likelihoods

        return next_event_combinations

    def create_cache(self, perfect_start_symbol=False):
        """ Creates a detect cache for a model that is conditioned on the previous attributes. """
        predictions = self.model.predict(self.dataset.features)
        number_attributes = self.dataset.num_attributes
        features = self.dataset.flat_features
        cache = dict()
        for case in range(self.dataset.num_cases):
            case_features = np.transpose(features[case])
            index_features = [np.zeros(self.dataset.max_len).astype(int) for _ in range(number_attributes)]
            # can stop after case length, do not need to continue caching for padding
            for index in range(self.dataset.case_lens[case]):
                for attribute in range(number_attributes):
                    index_features[attribute][index] = case_features[attribute][index:index + 1]
                    # last attribute at index
                    if attribute == number_attributes - 1:
                        # already next index
                        prediction_index = index + 2 if perfect_start_symbol else index + 1
                        index_prediction = predictions[0][case, prediction_index - 1:prediction_index, :]
                    else:
                        # current index
                        prediction_index = index + 1 if perfect_start_symbol else index
                        index_prediction = predictions[attribute + 1][case, prediction_index - 1:prediction_index, :]
                    # store after each attribute
                    index_key = walk_utils.transform_walk_to_dict_key(index_features, padded_walks=True)
                    cache[index_key] = index_prediction
        return cache

