
from percolation_improved import *

import networkx as nx
from networkx.readwrite import json_graph
import json


class K222WithDegree3Vertices:
    """
    Build graphs consisting of:
    - a K_{2,2,2} on 6 base vertices
    - additional vertices, each connected to exactly 3 existing vertices

    Base partition:
        A = [0, 1]
        B = [2, 3]
        C = [4, 5]
    """

    def __init__(self):
        self.G = nx.Graph()
        self.parts = {
            "A": [0, 1],
            "B": [2, 3],
            "C": [4, 5],
        }
        self._next_vertex = 6
        self._build_k222()

    def _build_k222(self):
        """Create the base K_{2,2,2}."""
        self.G = nx.complete_multipartite_graph(2,2,2)
        nx.set_node_attributes(self.G, "original", "type")

    def add_degree3_vertex(self, neighbors, rule=None):
        """
        Add one new vertex connected to exactly 3 existing vertices.

        Parameters
        ----------
        neighbors : iterable of length 3
            The 3 vertices to connect the new vertex to.

        rule : rule of addition, such as "AAB", saved in vertex data 'type'
        Returns
        -------
        int
            The label of the new vertex.
        """
        neighbors = list(neighbors)

        if len(neighbors) != 3:
            raise ValueError("A degree-3 vertex must have exactly 3 neighbors.")

        if len(set(neighbors)) != 3:
            raise ValueError("Neighbors must be distinct.")

        missing = [v for v in neighbors if v not in self.G]
        if missing:
            raise ValueError(f"These neighbors are not in the graph: {missing}")

        new_v = self._next_vertex
        self._next_vertex += 1

        self.G.add_node(new_v, type=rule)
        for u in neighbors:
            self.G.add_edge(new_v, u)

        return new_v

    def add_vertices_by_rule(self, n_new, rule, with_random=True):
        """
        Add n_new vertices according to a rule.
        Each vertex samples independently if with_random=True.
        """

        def idx():
            return random.randint(0, 1) if with_random else 0

        def get_neighbors(rule):
            presets = {
                "AAB": [self.parts["A"][0], self.parts["A"][1], self.parts["B"][idx()]],
                "AAC": [self.parts["A"][0], self.parts["A"][1], self.parts["C"][idx()]],
                "ABB": [self.parts["A"][idx()], self.parts["B"][0], self.parts["B"][1]],
                "ACC": [self.parts["A"][idx()], self.parts["C"][0], self.parts["C"][1]],
                "BBC": [self.parts["B"][0], self.parts["B"][1], self.parts["C"][idx()]],
                "BCC": [self.parts["B"][idx()], self.parts["C"][0], self.parts["C"][1]],
                "ABC": [
                    self.parts["A"][idx()],
                    self.parts["B"][idx()],
                    self.parts["C"][idx()],
                ],
            }

            if rule not in presets:
                raise ValueError(f"Unknown rule: {rule}")

            return presets[rule]

        assert rule is not None

        added = []
        for _ in range(n_new):
            neighbors = get_neighbors(rule)
            added.append(self.add_degree3_vertex(neighbors, rule))

        return added

    def add_vertices_by_rules(self, numbers_and_rules : tuple[tuple[int, str], ...]):
        """Params: numbers_and_rules: tuples of number of vertices and rule."""
        added = []
        for n_new, rule in numbers_and_rules:
            vertices = self.add_vertices_by_rule(n_new, rule)
            added.extend(vertices)

        return added

    def add_vertices_by_all_rules(self, vertices_to_add, use_ABC=False):
        """ Adds the same amount of vertices to all rules """
        all_rules = ["AAB", "ABB", "AAC",  "ACC", "BBC", "BCC"]
        if use_ABC:
            all_rules.append("ABC")

        return self.add_vertices_by_rules(tuple((vertices_to_add, rule) for rule in all_rules))

    def graph(self):
        """Return the underlying NetworkX graph."""
        return self.G

    def copy(self):
        """Return a copy of the graph."""
        return self.G.copy()

    def percolation_graph(self):
        return Graph(self.G)

    def transfer_edges_from_cut(self, num_edges=0, use_ABC=False):
        correspondence = {
            "AAB": "ABB",
            "ABB": "AAB",
            "AAC": "ACC",
            "ACC": "AAC",
            "BBC": "BCC",
            "BCC": "BBC",
        }

        base_types = set(correspondence.keys())
        allowed_types = base_types | ({"ABC"} if use_ABC else set())

        for _ in range(num_edges):
            # edges between original and allowed types
            cut_edges = [
                (u, v)
                for u, v in self.G.edges()
                if (
                           self.G.nodes[u].get("type") == "original"
                           and self.G.nodes[v].get("type") in allowed_types
                   )
                   or (
                           self.G.nodes[v].get("type") == "original"
                           and self.G.nodes[u].get("type") in allowed_types
                   )
            ]

            if not cut_edges:
                raise ValueError("No more cut edges to transfer.")

            u, v = random.choice(cut_edges)

            # x = non-original endpoint
            if self.G.nodes[u].get("type") == "original":
                original_vertex, x = u, v
            else:
                original_vertex, x = v, u

            x_type = self.G.nodes[x]["type"]

            # choose target types
            if use_ABC and x_type == "ABC":
                # ABC connects to a random other type (not ABC)
                target_types = list(base_types)
            else:
                # standard correspondence
                if x_type not in correspondence:
                    raise ValueError(f"Unsupported type for transfer: {x_type}")
                target_types = [correspondence[x_type]]

            # candidates of target types (excluding x)
            candidates = [
                w for w, data in self.G.nodes(data=True)
                if data.get("type") in target_types and w != x
            ]

            if not candidates:
                raise ValueError(f"No vertices of target type(s) {target_types}.")

            # prefer non-neighbors
            non_neighbors = [w for w in candidates if not self.G.has_edge(x, w)]
            if not non_neighbors:
                raise ValueError(
                    f"Vertex {x} of type {x_type} is already connected to all target vertices."
                )

            y = random.choice(non_neighbors)

            # transfer edge
            self.G.remove_edge(original_vertex, x)
            self.G.add_edge(x, y)

            original_type = self.G.nodes[original_vertex]["type"]
            y_type = self.G.nodes[y]["type"]

            print(f"removing {original_type} -> {x_type} and adding {x_type} -> {y_type}")
            print(f"removed edge: {original_vertex} -> {x}")
            print(f"added edge: {x} -> {y}")

def check_percolation(number_of_vertices_all_rules, use_ABC, num_edges_to_transfer=0):
    builder = K222WithDegree3Vertices()
    builder.add_vertices_by_all_rules(number_of_vertices_all_rules, use_ABC=use_ABC)
    builder.transfer_edges_from_cut(num_edges=num_edges_to_transfer, use_ABC=use_ABC)
    G = builder.percolation_graph()
    answer, final_graph = G.is_percolating(print_steps=False, return_final_graph=True)
    # PercolationGraph(build_dependencies_graph(final_graph)).print_graph()
    assert not answer

    # for v, data in final_graph.nodes(data=True):
    #     vertex_type = data["type"]
    #     if vertex_type == "original":
    #         continue  # Original K_{2,2,2} vertices can be connected to any type
    #
    #     if 'A' not in vertex_type and (final_graph.has_edge(v, 0) or final_graph.has_edge(v, 1)):
    #         raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to A vertices in the final graph.")
    #     if 'B' not in vertex_type and (final_graph.has_edge(v, 2) or final_graph.has_edge(v, 3)):
    #         raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to B vertices in the final graph.")
    #     if 'C' not in vertex_type and (final_graph.has_edge(v, 4) or final_graph.has_edge(v, 5)):
    #         raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to C vertices in the final graph.")
    #
    # print('Conjecture holds.')
    save_graph_to_json(final_graph, f"final_graph_{number_of_vertices_all_rules}_vertices_per_rule.json")


def check_from_computed_graph(num_vertices):
    G = read_graph_from_json(f'final_graph_{num_vertices}_vertices_per_rule.json')
    print(f'There are {len(list(nx.connected_components(nx.complement(G))))} connected components')
    for component in nx.connected_components(nx.complement(G)):
        H = nx.complement(G).subgraph(component)
        n = H.number_of_nodes()
        # assert H.number_of_edges() == n * (n - 1) // 2

    for v, data in G.nodes(data=True):
        print(f'vertex {v} of type {data['type']} has degree {G.degree(v)}')

    PercolationGraph(G).print_graph()

def build_dependencies_graph(G : nx.Graph):
    H = nx.Graph()
    for v, data in G.nodes(data=True):
        for u in G.neighbors(v):
            neighbor_type = G.nodes[u].get("type")
            H.add_edge(data['type'], neighbor_type)

    return H

if __name__ == "__main__":
    num_vertices = 2
    check_percolation(num_vertices, use_ABC=True, num_edges_to_transfer=1)
    check_from_computed_graph(num_vertices)
    # print(f'check: {4 * (num_vertices + 1)}')


