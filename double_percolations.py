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

    def is_double_percolating(self, return_final_graph=False):
        graph = self.copy()
        while True:
            graph_check = graph.copy()
            answer, graph = graph.is_k5_percolating(return_final_graph=True)
            if answer:
                graph = PercolationGraph(nx.complete_graph(self.number_of_nodes()))
                break
            if graph_check.number_of_edges() < graph.number_of_edges():
                continue

            answer, graph = graph.is_k222_percolating(return_final_graph=True)
            if answer:
                graph = PercolationGraph(nx.complete_graph(self.number_of_nodes()))
                break
            if graph_check.number_of_edges() == graph.number_of_edges():
                break
        if return_final_graph:
            return graph.number_of_edges() == self.number_of_nodes() * (self.number_of_nodes() - 1) // 2, graph
        return graph.number_of_edges() == self.number_of_nodes() * (self.number_of_nodes() - 1) // 2



if __name__ == "__main__":
    for i, (a, b, c) in enumerate(itertools.combinations([1,2,3,4,5], 3)):
        G = DoublePercolation(nx.complete_multipartite_graph(2, 2, 2))
        print(f'Processing {i}\'th iteration, a,b,c={a,b,c}')
        new_node = max(G.nodes()) + 1
        G.add_node(new_node)
        G.add_edges_from([(new_node, a), (new_node, b), (new_node, c)])
        answer, final_graph = G.is_double_percolating(return_final_graph=True)
        if final_graph.has_edge(0,6):
            G.remove_node(6)
            answer = G.is_double_percolating()
            if not answer:
                print('Found counterexample:')
                print(a,b,c)



