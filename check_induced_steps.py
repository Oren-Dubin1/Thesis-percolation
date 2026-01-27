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
from percolation_improved import run_percolation_experiments
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

    G = nx.complete_graph(n)
    for edge, wit in zip(order_of_additions, wits):
        is_induced = is_induced_step(G, wit)
        if is_induced:
            extended = join_incrementally(G, wit)
            if not subgraph_has_outside_vertex_of_degree_at_least(G, extended):
                # No nice 4-degree extension found
                PercolationGraph(G).print_graph()
                print(edge, wit, extended)
                raise "graph is not locally rigid"

        G.remove_edge(*edge)





if __name__ == "__main__":
    n = 12
    graphs = read_graphs_from_edgelist(f'percolating Graphs/n_{n}')[27:28]
    for i, G in enumerate(graphs):
        print(f"Checking graph {i+1}/{len(graphs)}")
        check_if_graph_can_be_locally_fixed(G)


