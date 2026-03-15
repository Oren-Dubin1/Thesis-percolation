import itertools

import networkx as nx

from percolation_improved import Graph


def get_all_cuts_of_sizes_3_4(sizes=None, as_list=True, connected_only=False):
    """Generate bipartite cut graphs for given partition sizes.

    Parameters:
        sizes (iterable of (int,int) tuples, optional): list of (left,right) sizes to
            generate. Defaults to [(3,3),(3,4),(4,3),(4,4)].
        as_list (bool): If True, return a list of (left,right,G) tuples. Otherwise return a
            generator that yields (left,right,G) lazily.
        connected_only (bool): If True, only include connected graphs.

    Yields or returns:
        (left, right, G) tuples where G is a networkx.Graph with nodes 0..(left+right-1),
        left part = 0..left-1, right part = left..left+right-1.
    """

    if sizes is None:
        sizes = [(3, 3), (3, 4), (4, 3), (4, 4)]

    def generate():
        for left, right in sizes:
            n_left = left
            n_right = right
            total_nodes = n_left + n_right
            left_nodes = list(range(0, n_left))
            right_nodes = list(range(n_left, n_left + n_right))
            edge_positions = [(u, v) for u in left_nodes for v in right_nodes]
            max_mask = 1 << (len(edge_positions))
            for mask in range(max_mask):
                G = nx.Graph()
                G.add_nodes_from(range(total_nodes))
                for bit_idx, (u, v) in enumerate(edge_positions):
                    if (mask >> bit_idx) & 1:
                        G.add_edge(u, v)

                # Connect left, right to a clique
                for u in range(left):
                    for v in range(left):
                        if u != v:
                            G.add_edge(u, v)
                for u in range(left, left + right):
                    for v in range(left, left + right):
                        if u != v:
                            G.add_edge(u, v)

                if connected_only and not nx.is_connected(G):
                    continue
                yield left, right, G

    gen = generate()
    if as_list:
        return list(gen)
    return gen

def get_all_cliques_connected(get_cuts=False):
    cuts = get_all_cuts_of_sizes_3_4(as_list=True, connected_only=True)[46400:]
    graphs = []
    for cut in cuts:
        left, right, G = cut
        # Add a K_4 to left and a K_4 to right, totally adding 8 vertices
        # Make it on nodes 10..13 for left and 14..17 for right to avoid overlap with the cut vertices
        G.add_edges_from(itertools.combinations(range(10, 14), 2))
        G.add_edges_from(itertools.combinations(range(14, 18), 2))
        # connect the K_4's to the cut vertices
        for u in range(left):
            for v in range(10, 14):
                G.add_edge(u, v)
        for u in range(left, left + right):
            for v in range(14, 18):
                G.add_edge(u, v)

        if get_cuts:
            graphs.append((cut, G))
        else:
            graphs.append(G)
    return graphs

if __name__ == "__main__":
    n = 13
    # run_percolation_experiments(n=n, max_tries=10000, output_dir='Double percolating Graphs', double_percolation=True)
    # graph = read_graphs_from_edgelist(f'percolating Graphs/n_{n}')[0]
    # result, wits = graph.is_percolating(k_222_plus=False, document_steps=True)
    # print(f'Percolated: {result}, Witnesses: {wits}')

    graphs = get_all_cliques_connected(get_cuts=True)
    i = 1
    for cut, graph in graphs:
        if i % 50 == 0:
            print(f'Testing graph {i}/{len(graphs)}, passed {i/len(graphs)*100:.2f}%')
        i += 1
        left, right, H = cut
        H = Graph(H)
        if H.is_percolating():
            continue
        G = Graph(graph)
        result = G.is_percolating()
        if result:
            print('Found a graph with non-percolating boundary that percolates!')
            print(f'Left size: {left}, Right size: {right}')
            print('Cut graph edges:')
            print(H.graph.edges())
            print('Full graph edges:')
            print(G.graph.edges())
            raise AssertionError("Found a graph with non-percolating boundary that percolates!")

    print("All graphs with non-percolating boundary are not percolating.")
