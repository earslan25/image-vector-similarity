import torch
import numpy as np


class QueryComparator:
    def __init__(self, queries):
        # assumes queries is already transformed into vectors
        self.queries = queries

    # assumes main_input is already transformed into a vector
    def compare(self, main_input, threshold):
        # main_input: the input to compare to the queries
        # return: the indices of the queries that are similar to the main_input
        # TODO use an open loop to show progress
        main_input = main_input.squeeze(0)
        print("comparing...")
        print(self.queries.shape)
        print(main_input.shape)
        return np.array([i for i, query in enumerate(self.queries)
                         if torch.nn.functional.cosine_similarity(main_input, query, dim=0) >= threshold])
