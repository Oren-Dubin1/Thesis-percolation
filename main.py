import networkx as nx
import itertools
from networkx.generators.atlas import graph_atlas_g
import sys
import contextlib


def is_k222_minus_subgraph(supergraph, missing_edge, node_set):
    nodes = list(node_set)
    for A in itertools.combinations(nodes, 2):
        rest = [v for v in nodes if v not in A]
        for B in itertools.combinations(rest, 2):
            C = [v for v in rest if v not in B]
            cross_edges = set()
            cross_edges.update(itertools.product(A, B))
            cross_edges.update(itertools.product(A, C))
            cross_edges.update(itertools.product(B, C))
            cross_edges = set(tuple(sorted(e)) for e in cross_edges)

            if tuple(sorted(missing_edge)) not in cross_edges:
                continue
            required_edges = cross_edges - {tuple(sorted(missing_edge))}
            if all(supergraph.has_edge(*e) for e in required_edges):
                return True
    return False


def is_k222_percolating(G):
    G = G.copy()
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    def all_edges(nodelist):
        return set(tuple(sorted(e)) for e in itertools.combinations(nodelist, 2))

    changed = True
    while changed:
        changed = False
        current_edges = set(tuple(sorted(e)) for e in G.edges())
        missing_edges = all_edges(nodes) - current_edges

        for u, v in missing_edges:
            candidate = G.copy()
            candidate.add_edge(u, v)

            for S in itertools.combinations(nodes, 6):
                if u not in S or v not in S:
                    continue
                if is_k222_minus_subgraph(candidate, (u, v), S):
                    G.add_edge(u, v)
                    changed = True
                    break
            if changed:
                break

    return G.number_of_edges() == n * (n - 1) // 2


def print_graph(G, file=None):
    for i in range(len(G.nodes())):
        print(i, file=file)
    for e in G.edges():
        print(*e,file=file)


def check_vertex_graphs_with_degree3_extension(num_edges=6):
    with open('out1.txt', 'w') as f:
        graphs = graph_atlas_g()
        G6 = [nx.convert_node_labels_to_integers(G) for G in graphs if G.number_of_nodes() == num_edges]
        G6 = [G for G in G6 if G.number_of_edges() >= 3 * num_edges-6]  # Based on the conjecture
        total = 0
        percolating = 0

        for i, G in enumerate(G6):
            base_nodes = list(G.nodes())

            for neighbor_set in itertools.combinations(base_nodes, 3):
                G_ext = G.copy()
                new_node = num_edges
                # print(f'new_node={new_node}')
                G_ext.add_node(new_node)
                for v in neighbor_set:
                    G_ext.add_edge(new_node, v)

                total += 1
                if is_k222_percolating(G_ext):
                    percolating += 1
                    # Check if G without the added vertex is percolating
                    G_ext.remove_node(new_node)
                    if not is_k222_percolating(G_ext):
                        print('--------------------------------------------', file=f)
                        G_ext.add_node(new_node)
                        for v in neighbor_set:
                            G_ext.add_edge(new_node, v)
                        print_graph(G_ext, f)

            if i % 10 == 0:
                print(f"Processed {i+1}/{len(G6)} base graphs...")

        print(f"\nTotal 7-vertex graphs (degree-3 extension): {total}")
        print(f"Percolating graphs: {percolating}")
        print(f"Non-percolating graphs: {total - percolating}")



if __name__ == '__main__':
    check_vertex_graphs_with_degree3_extension(7)


# if __name__ == '__main__':
#     check_vertex_graphs_with_degree3_extension(7)
