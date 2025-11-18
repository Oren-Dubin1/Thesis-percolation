import networkx as nx
import itertools


class Percolation:
    def is_k5_minus_subgraph(self, subgraph, missing_edge):
        if subgraph.number_of_nodes() < 5:
            return False
        nodes = list(subgraph.nodes())
        for subset in itertools.combinations(nodes, 5):
            sub = subgraph.subgraph(subset)
            if sub.number_of_edges() == 9 and not sub.has_edge(*missing_edge):
                return True
        return False


    def is_k5_percolating(self, G):
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

                for S in itertools.combinations(nodes, 5):
                    if u not in S or v not in S:
                        continue
                    candidate.remove_edge(u, v)
                    if self.is_k5_minus_subgraph(candidate, (u, v)):
                        G.add_edge(u, v)
                        changed = True
                        break
                if changed:
                    break

        return G.number_of_edges() == n * (n - 1) // 2


    def parse_graphs_from_file(self, path):
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


class TestPercolating:
    def __init__(self):
        pass

    def test_is_k5_percolating(self):
        # Test 1: K5 minus one edge – should percolate
        G1 = nx.complete_graph(5)
        G1.remove_edge(0, 1)
        assert self.is_k5_percolating(G1) == True, "Test 1 Failed: K5^- should percolate"

        # Test 2: Disconnected graph – should not percolate
        G2 = nx.Graph()
        G2.add_nodes_from(range(5))
        assert self.is_k5_percolating(G2) == False, "Test 2 Failed: Empty graph shouldn't percolate"

        # Test 3: K5 – already complete
        G3 = nx.complete_graph(5)
        assert self.is_k5_percolating(G3) == True, "Test 3 Failed: K5 is trivially percolating"

        G4 = nx.complete_graph(5)
        G4.remove_edge(0, 1)
        G4.add_node(5)
        G4.add_edge(0, 5)
        G4.add_edge(1, 5)
        G4.add_edge(2, 5)
        G4.add_edge(3, 5)

        assert self.is_k5_percolating(G4), "Test 4 Failed: G4 is percolating"

        print("Basic unit tests passed ✅")

    def test_graph_parsing(self):
        mock_input = """--------------------------------------------
0
1
2
3
4
5
0 1
0 2
0 3
1 2
1 3
2 3
3 4
4 5
--------------------------------------------
0
1
2
3
0 1
1 2
2 3
        """
        # Mock the file
        with open("test_input.txt", "w") as f:
            f.write(mock_input)

        graphs = self.parse_graphs_from_file("test_input.txt")
        assert len(graphs) == 2, "Parsing Failed: Expected 2 graphs"

        g0 = graphs[0]
        assert g0.number_of_nodes() == 6
        assert g0.has_edge(0, 1)
        assert g0.has_edge(4, 5)

        g1 = graphs[1]
        assert g1.number_of_nodes() == 4
        assert g1.has_edge(2, 3)

        print("Graph parsing tests passed ✅")

    def check(self):
        self.test_is_k5_percolating()
        self.test_graph_parsing()



def main():
    perc = Percolation()
    graphs = perc.parse_graphs_from_file('7nodes1special.txt')
    for i, G in enumerate(graphs):
        max_node = max(G.nodes())
        G.remove_node(max_node)
        result = perc.is_k5_percolating(G)
        print(f"Graph {i+1}: K5-percolating after removing node {max_node}? {result}")


if __name__ == '__main__':
    main()

