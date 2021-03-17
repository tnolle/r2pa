import numpy as np


class NextEventPredictor:

    # TODO: handle next event threshold differently?
    def __init__(self, dataset, model, next_event_threshold, use_cache, name=None):
        self.dataset = dataset
        self.model = model  # can be a graph or neural network for instance
        self.next_event_threshold = next_event_threshold
        self.next_event_coverage = 0.85
        self.use_cache = use_cache
        self.cache = None
        self.uncached_cases = []
        self.name = self.model.name if name is None else name

    def next_event(self, case, case_length):
        pass

    def next_event_with_cache(self, case, case_length):
        pass

    def get_next_events(self, case, case_length):  # sequence length without padding, not max_len
        if self.use_cache:
            return self.next_event_with_cache(self, case, case_length)
        return self.next_event(self, case, case_length)

    def event_from_prediction_by_threshold(self, next_event_predictions):
        """ Given a one-dimensional array of likelihoods (summing up to 1),
            select the next events based on a threshold, i.e. cut off at this threshold.
            The returned events are the indices of the entries satisfying the threshold.
            :return: (events, likelihoods) """
        next_event_attributes = np.where(next_event_predictions >= self.next_event_threshold)[0]

        if len(next_event_attributes) == 0:
            print("WARNING: No next event for current threshold. Taking the most likely next attribute value.")
            next_event_attributes = [np.argmax(next_event_predictions)]

        next_event_likelihoods = np.take(next_event_predictions, next_event_attributes)
        return next_event_attributes, next_event_likelihoods

    def event_from_prediction_by_coverage(self, next_event_predictions):
        sorted_predictions = np.flip(np.sort(next_event_predictions))
        cumulative_sum_predictions = np.cumsum(sorted_predictions)
        coverage_index = np.argmax(cumulative_sum_predictions >= self.next_event_coverage)
        covered_likelihoods = sorted_predictions[0:coverage_index+1]
        covered_attributes = np.zeros(covered_likelihoods.shape)
        for index, likelihood in enumerate(covered_likelihoods):
            covered_attributes[index] = np.argmax(next_event_predictions == likelihood)
        return covered_attributes, covered_likelihoods

    def create_cache(self):
        pass

    def add_to_cache(self, key, prediction):
        """ Adds a new prediction to the cache. """
        self.arguments.cache[key] = prediction

    def get_uncached_cases(self):
        """ Removes all cases that are subcases of other cases as well as duplicates. """
        # remove 'prefixes'/subset cases
        filtered_cases = []
        for x in self.uncached_cases:
            length_x = len(x)
            x_in_w = False
            # check if x is in any case
            for w in self.uncached_cases:
                # must be shorter, cannot be sublist otherwise
                if length_x >= len(w):
                    continue
                # know that x is shorter than w
                sublist_w = w[:length_x]
                if x == sublist_w:
                    x_in_w = True
                    break
            if not x_in_w:
                filtered_cases.append(x)
        return set(filtered_cases)  # filter out duplicates

    def clear_cached_cases(self):
        self.uncached_cases = []
