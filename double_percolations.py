import networkx as nx
import itertools

import numpy as np
from Graphs import PercolationGraph
from main import is_k222_percolating

class DoublePercolation(PercolationGraph):
    def __init__(self, base_graph=None, **kwargs):
        super().__init__(**kwargs)
        if base_graph is not None:
            self.update(base_graph)

    def is_double_percolating(self):
        graph = self.copy()
        while True:
            graph_check = graph.copy()
            answer, graph = graph.is_k5_percolating(return_final_graph=True)
            if graph_check.number_of_edges() < graph.number_of_edges():
                continue

            answer, graph = graph.is_k222_percolating(return_final_graph=True)
            if graph_check.number_of_edges() == graph.number_of_edges():
                break
        return graph.number_of_edges() == self.number_of_nodes() * (self.number_of_nodes() - 1) // 2




