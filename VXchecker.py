import networkx as nx
import itertools
import numpy as np
from Graphs import PercolationGraph


class VXChecker:
    def __init__(self,
                 graph : PercolationGraph = None,
                 v=0,
                 x=-1):
        self.graph = graph
        self.v = v
        self.x = x

    def check_edge_can_be_added(self, return_vertex=False):
        """Check if there exists a K_{2,2,2} including the maximum node, v,x."""
        n = self.graph.number_of_nodes()
        if n < 6:
            return (False, None) if return_vertex else False

        nodes = list(self.graph.nodes())
        max_node = max(nodes)

        other_nodes = [node for node in nodes if node not in [max_node, self.v, self.x]]

        for node_combination in itertools.combinations(other_nodes, 3):
            candidate_nodes = list(node_combination) + [max_node] + [self.x] + [self.v]
            subgraph = self.graph.subgraph(candidate_nodes)
            for n in node_combination:
                if not subgraph.has_edge(max_node, n) and \
                       subgraph.is_k222_minus_subgraph((max_node, n), candidate_nodes):
                    if return_vertex:
                        return True, n
                    return True
        if return_vertex:
            return False, None
        return False


    def enumerate_possible_edges(self, num_additions=1):
        Base = PercolationGraph(nx.complete_multipartite_graph(2, 2, 2))
        v = 0
        x = 1
        G = Base.copy()
        for num_neighbors in range(3, G.number_of_nodes()+1):
            for _ in range(num_additions):
                neighbors = list(itertools.combinations(G.nodes, num_neighbors))
                G = Base.copy()
                G.add_edges_from([(v, n) for n in neighbors])








if __name__ == '__main__':
    VXChecker().enumerate_possible_edges(num_additions=1)



