import itertools
import networkx as nx
import random

import numpy as np
from networkx.classes import non_edges

from Graphs import PercolationGraph
import os
import matplotlib.pyplot as plt
from double_percolations import DoublePercolation
import os
from percolation_improved import run_percolation_experiments, Graph
from percolation_improved import read_graphs_from_edgelist

def is_induced_step(graph: nx.Graph, witness) -> bool:
    # Check if the subgraph induced by the witness is a K222 or has more edges
    subgraph = graph.subgraph(witness)
    return subgraph.number_of_edges() == 12

def subgraph_has_outside_vertex_of_degree_at_least(graph: nx.Graph, nodes, degree: int=4) -> bool:
    # check if the subgraph on nodes has an outside vertex connected to at least 'degree' nodes in the subgraph
    S = set(nodes)

    return any(
        v not in S and sum(1 for u in graph[v] if u in S) >= degree
        for v in graph.nodes()
    )

def join_all_deg_3_to_subgraph(graph: nx.Graph, nodes):
    # do this, but return an iterator
    S = set(nodes)
    # find all vertices of degree 3 into S, not in S
    deg_3_vertices = [
        v for v in graph.nodes()
        if v not in S and sum(1 for u in graph[v] if u in S) == 3
    ]
    return nodes + tuple(deg_3_vertices)

def join_incrementally(graph: nx.Graph, nodes):
    S = set(nodes)
    while True:
        added = False
        for v in graph.nodes():
            if v not in S:
                deg = sum(1 for u in graph[v] if u in S)
                if deg >= 4:
                    break

        for v in graph.nodes():
            if v not in S:
                deg = sum(1 for u in graph[v] if u in S)
                if deg == 3:
                    S.add(v)
                    added = True
        if not added:
            break

    return tuple(S)

def check_if_graph_can_be_locally_fixed(graph: nx.Graph) -> None:
    result, order_of_additions, wits = graph.is_percolating(document_steps=True)

    order_of_additions = list(reversed(order_of_additions))
    wits = list(reversed(wits))
    # print(order_of_additions)
    G = nx.complete_graph(n)
    number_ind_edges = 0
    for edge, wit in zip(order_of_additions, wits):
        if tuple(sorted(edge)) in [(4,8), (0,1), (2,10)]:
            number_ind_edges += 1

        if number_ind_edges == 3:
            PercolationGraph(G).print_graph()
            print(edge, wit)
            return
        is_induced = is_induced_step(G, wit)
        if is_induced:
            extended = join_incrementally(G, wit)
            if not subgraph_has_outside_vertex_of_degree_at_least(G, extended):
                # No nice 4-degree extension found
                PercolationGraph(G).print_graph()
                print(edge, wit, extended)
                raise "graph is not locally rigid"

        G.remove_edge(*edge)

def subgraph_is_clique(graph: nx.Graph, nodes) -> bool:
    subgraph = graph.subgraph(nodes)
    n = len(nodes)
    return subgraph.number_of_edges() == n * (n - 1) // 2


def check_conj_k222_clique(attempts=100):
    """ Check Conjecture: the graph is a K_{2,2,2} and a clique (K7 for example), with the cut undecided.
        is it true that after removing an edge from the octahedron, the rigidity rank does not change?"""


    G = nx.complete_multipartite_graph(2,2,2)
    K222_edges = list(G.edges())
    K = nx.complete_graph(9)
    # Rename K to be disjoint from G
    K = nx.relabel_nodes(K, {i: i + 6 for i in K.nodes()}, copy=False)
    H = nx.compose(G, K)

    edge_in_cut = [(u,v) for u in range(6) for v in range(6,13)]

    for i in range(attempts):
        print("Attempt", i+1)
        while True:
            number_of_edges_to_add = random.randint(4, len(edge_in_cut))
            edges_to_add = random.sample(edge_in_cut, number_of_edges_to_add)
            print("Adding edges:", len(edges_to_add))
            H_temp = H.copy()
            H_temp.add_edges_from(edges_to_add)
            graph_obj = Graph(H_temp)
            result, graph = graph_obj.is_percolating(return_final_graph=True)
            if subgraph_is_clique(graph, range(6)):
                break

        __ , rank = graph_obj.is_rigid(return_rank=True)
        for edge in K222_edges:
            H_modified = H_temp.copy()
            H_modified.remove_edge(*edge)
            graph_modified = Graph(H_modified)
            __, modified_rank = graph_modified.is_rigid(return_rank=True)
            if rank != modified_rank:
                print("Conjecture failed!")
                print("Removed edge:", edge)
                print("Original rank:", rank)
                print("Modified rank:", modified_rank)
                PercolationGraph(H_temp).print_graph()
                PercolationGraph(H_modified).print_graph()
                return



if __name__ == "__main__":
    check_conj_k222_clique(attempts=1000)




