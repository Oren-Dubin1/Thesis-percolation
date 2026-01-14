import itertools
import unittest

import networkx as nx
import numpy as np
from jinja2.nodes import Assign

import main
from Graphs import PercolationGraph
from main import print_graph, is_k222_percolating
import Graphs


class CreateLargeGraph:
    def __init__(self, n: int, init_graph: Graphs.PercolationGraph):
        self.n = n
        self.graph = PercolationGraph(init_graph)  # Must be K222 percolating


    def enlarge(self):
        while self.graph.number_of_nodes() < self.n:
            print(f'current number of nodes={self.graph.number_of_nodes()}')
            # Choose a removable edge
            edges = list(self.graph.edges())
            u, v = edges[np.random.randint(len(edges))]

            self.graph.remove_edge(u, v)

            # Try to find two other nodes a, b such that adding w to u,v,a,b creates a K_{2,2,2}^-
            candidate_nodes = set(self.graph.nodes()) - {u, v}
            found = False

            for a, b in itertools.combinations(candidate_nodes, 2):
                # Add temporary node w with edges to u, v, a, b
                w = max(self.graph.nodes()) + 1
                self.graph.add_node(w)
                self.graph.add_edges_from([(w, u), (w, v), (w, a), (w, b)])

                if self.graph.is_k222_percolating():
                    found = True
                    break  # Success, graph is now enlarged

                # Otherwise, undo and try next
                self.graph.remove_node(w)

            if not found:
                # If no valid (a, b) worked, restore the edge and try another iteration
                self.graph.add_edge(u, v)


    def check_conjecture_3n_6(self, edge):
        if self.graph.number_of_edges() > 3 * self.graph.number_of_nodes() - 6:
            return

        answer = self.graph.is_k_222_percolating_without_edge(edge)
        if answer:
            self.graph.remove_edge(*edge)
            self.graph.print_graph()
            raise "Found percolating graph on < 3n-6 edges."


    @staticmethod
    def get_opposite(subgraph, node, edge_to_add=None):
        assert subgraph.number_of_nodes() == 6

        comp = nx.complement(subgraph)
        if edge_to_add: comp.remove_edge(*edge_to_add)
        neighbor = next(comp.neighbors(node))
        return neighbor

    def decide_vertices_to_connect_one_outer_one_inner(self, subgraph, u,v,x,y):
        # Assuming u is outer
        assert subgraph.has_edge(u, x) and subgraph.has_edge(u, y)  # u is outer
        if subgraph.has_edge(v, x):
            node = x
        elif subgraph.has_edge(v, y):
            node = y

        else:
            subgraph.print_graph()
            raise AssertionError

        return u, node, self.get_opposite(subgraph, u), self.get_opposite(subgraph, node)


    def decide_vertices_to_connect_both_inner(self, subgraph, u,v,x,y):
        outer = list(set(range(6)) - {u,v,x,y})
        return u, self.get_opposite(subgraph, u), outer[0], outer[1]

    def decide_vertices_to_connect_special_outer(self, subgraph, u,v,x,y):
        return u,v, self.get_opposite(subgraph, u), self.get_opposite(subgraph, v)

    def decide_vertices_to_connect_special_inner(self, subgraph, u,v,x,y):
        if v != x and v != y:  # v is inner
            u,v = v,u
        # u is inner
        return tuple(subgraph.nodes - {u, self.get_opposite(subgraph, u)})


    def decide_vertices_to_connect(self, edge):
        # By assumption - G-e does not percolate. G does percolate
        assert edge in self.graph.edges()
        graph = self.graph.copy()
        graph.remove_edge(*edge)
        subgraph, f = graph.find_k222_without_two_edges(edge, induced=False)  # Problem lies here.
        # if subgraph is None:  # No K222^{--} in the graph with edge being one of the missing edges
        #     print(edge)
        #     graph.print_graph()

        assert subgraph is not None
        assert f is not None
        u,v = edge
        x,y = f
        # assert u != x and u != y and v != x and v != y
        # First case in the proof - (u,v)
        assert subgraph.degree[x] == 3 and subgraph.degree[y] == 3
        if subgraph.degree[u] == 4 and subgraph.degree[v] == 4:
            if subgraph.has_edge(u, x) and subgraph.has_edge(u,y): # u is outer
                return self.decide_vertices_to_connect_one_outer_one_inner(subgraph, u,v,x,y)

            elif subgraph.has_edge(v, x) and subgraph.has_edge(v,y): # v is outer
                return self.decide_vertices_to_connect_one_outer_one_inner(subgraph, v,u,x,y)

            else:
                return self.decide_vertices_to_connect_both_inner(subgraph, u,v,x,y)

        else:
            if subgraph.has_edge(v, x) and subgraph.has_edge(v,y) or subgraph.has_edge(u, x) and subgraph.has_edge(u, y):
                return self.decide_vertices_to_connect_special_outer(subgraph, u,v,x,y)
            else:
                return self.decide_vertices_to_connect_special_inner(subgraph, u,v,x,y)

    def test_conjecture(self):
        assert self.graph.is_k222_percolating()
        if not self.graph.is_k5_percolating():
            print("Found graph which is K222 percolating and not K5 percolating.")
            self.graph.print_graph()
            raise AssertionError
        return True

    def test_rigid(self):
        if not self.graph.is_rigid():
            print("Found graph which is not rigid.")
            self.graph.print_graph()
            raise AssertionError
        return True



    def smart_enlarge(self):
        while self.graph.number_of_nodes() < self.n:
            print(f'current number of nodes={self.graph.number_of_nodes()}')

            assert self.graph.number_of_edges() == 3 * self.graph.number_of_nodes() - 6

            # self.test_conjecture()
            self.test_rigid()

            edges = list(self.graph.edges())
            while True:
                idx = np.random.randint(len(edges))
                edge_to_remove = edges[idx]
                # edge_to_remove = edges[1]  #ONLY FOR TESTING

                # self.check_conjecture_3n_6(edge_to_remove)  # Along the way check

                try:
                    vertices_to_connect = self.decide_vertices_to_connect(edge_to_remove)
                except AssertionError:
                    continue
                # print("Before:")
                # self.graph.print_graph()
                self.graph.remove_edge(*edge_to_remove)
                # print('removing', edge_to_remove)

                new_node = max(self.graph.nodes) + 1
                self.graph.add_node(new_node)
                self.graph.add_edges_from([(new_node, vertex) for vertex in vertices_to_connect])

                if self.graph.is_k222_percolating():
                    break

                # print("After:")
                # self.graph.print_graph()

if __name__ == "__main__":
    init_graph = PercolationGraph(nx.complete_multipartite_graph(2,2,2))
    creator = CreateLargeGraph(20, init_graph)
    creator.smart_enlarge()
    print("Final graph:")
    creator.graph.print_graph()
