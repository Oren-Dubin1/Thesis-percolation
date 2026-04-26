
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
        all_rules = ["AAB", "AAC", "ABB", "ACC", "BBC", "BCC"]
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


def check_percolation(number_of_vertices_all_rules, use_ABC):
    builder = K222WithDegree3Vertices()
    builder.add_vertices_by_all_rules(number_of_vertices_all_rules, use_ABC=use_ABC)
    G = builder.percolation_graph()
    answer, final_graph = G.is_percolating(print_steps=True, return_final_graph=True)
    assert not answer

    for v, data in final_graph.nodes(data=True):
        vertex_type = data["type"]
        if vertex_type == "original":
            continue  # Original K_{2,2,2} vertices can be connected to any type

        if 'A' not in vertex_type and (final_graph.has_edge(v, 0) or final_graph.has_edge(v, 1)):
            raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to A vertices in the final graph.")
        if 'B' not in vertex_type and (final_graph.has_edge(v, 2) or final_graph.has_edge(v, 3)):
            raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to B vertices in the final graph.")
        if 'C' not in vertex_type and (final_graph.has_edge(v, 4) or final_graph.has_edge(v, 5)):
            raise AssertionError(f"Vertex {v} of type {data['type']} should not be connected to C vertices in the final graph.")

    print('Conjecture holds.')
    save_graph_to_json(final_graph, f"final_graph_{number_of_vertices_all_rules}_vertices_per_rule.json")


def check_from_computed_graph(num_vertices):
    G = read_graph_from_json(f'final_graph_{num_vertices}_vertices_per_rule.json')

    for v, data in G.nodes(data=True):
        print(f'vertex of type {data['type']} has degree {G.degree(v)}')
        vertex_type = data["type"]
        if vertex_type == "original":
            continue

def build_dependencies_graph(G : nx.Graph):
    H = nx.Graph()
    for v, data in G.nodes(data=True):
        for u in G.neighbors(v):
            neighbor_type = G.nodes[u].get("type")
            H.add_edge(data['type'], neighbor_type)

    return H

if __name__ == "__main__":
    num_vertices = 5
    check_percolation(num_vertices, use_ABC=True)
    check_from_computed_graph(num_vertices)
    print(f'check: {4 * (num_vertices + 1)}')


