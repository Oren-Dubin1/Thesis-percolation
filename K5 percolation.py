import networkx as nx
import itertools


def is_k5_minus_subgraph(subgraph, missing_edge):
    if subgraph.number_of_nodes() < 5:
        return False
    nodes = list(subgraph.nodes())
    for subset in itertools.combinations(nodes, 5):
        sub = subgraph.subgraph(subset)
        if sub.number_of_edges() == 9 and not sub.has_edge(*missing_edge):
            return True
    return False


def is_k5_percolating(G):
    G = G.copy()
    nodes = list(G.nodes())
    changed = True
    while changed:
        changed = False
        missing_edges = set(itertools.combinations(nodes, 2)) - set(G.edges()) - set((v,u) for (u,v) in G.edges())
        for u, v in missing_edges:
            candidate = G.copy()
            candidate.add_edge(u, v)
            for S in itertools.combinations(nodes, 5):
                if u not in S or v not in S:
                    continue
                if is_k5_minus_subgraph(candidate.subgraph(S), (u, v)):
                    G.add_edge(u, v)
                    changed = True
                    break
            if changed:
                break
    return G.number_of_edges() == len(nodes) * (len(nodes) - 1) // 2


def parse_graphs_from_file(path):
    graphs = []
    with open(path, 'r') as f:
        lines = f.read().split('--------------------------------------------')
        for block in lines:
            block = block.strip()
            if not block:
                continue
            lines = block.split('\n')
            num_nodes = 0
            edges = []
            for line in lines:
                if line.strip() == '':
                    continue
                if ' ' not in line:
                    num_nodes += 1
                else:
                    u, v = map(int, line.strip().split())
                    edges.append((u, v))
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges)
            graphs.append(G)
    return graphs


def main():
    graphs = parse_graphs_from_file('7nodes1special.txt')
    for i, G in enumerate(graphs):
        if i == 1:
            print(G)
        max_node = max(G.nodes())
        G.remove_node(max_node)
        result = is_k5_percolating(G)
        print(f"Graph {i+1}: K5-percolating after removing node {max_node}? {result}")


if __name__ == '__main__':
    main()
