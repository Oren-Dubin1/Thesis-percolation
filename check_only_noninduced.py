from percolation_improved import *


if __name__ == "__main__":
    n = 8
    for G in iter_graphs_for_n(n):
        g = Graph(G)
        if not g.is_percolating():
            answer, graph = g.is_percolating(return_final_graph=True)
            PercolationGraph(graph).print_graph()
            raise Exception("Found a non-percolating graph")
