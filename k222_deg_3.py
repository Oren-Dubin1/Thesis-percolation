
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

    def transfer_edges_from_cut(self, num_edges=0, use_ABC=False, force_ABC_connect_to_ABC=False):
        if force_ABC_connect_to_ABC and not use_ABC:
            raise ValueError("force_ABC_connect_to_ABC requires use_ABC=True")

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

            if self.G.nodes[u].get("type") == "original":
                original_vertex, x = u, v
            else:
                original_vertex, x = v, u

            x_type = self.G.nodes[x]["type"]

            if use_ABC and x_type == "ABC":
                if force_ABC_connect_to_ABC:
                    target_type = "ABC"
                else:
                    target_type = random.choice(list(base_types | {"ABC"}))
            else:
                if x_type not in correspondence:
                    raise ValueError(f"Unsupported type for transfer: {x_type}")

                # Equal chance: itself or corresponding type
                target_type = random.choice([x_type, correspondence[x_type], 'ABC'])

            candidates = [
                w for w, data in self.G.nodes(data=True)
                if data.get("type") == target_type and w != x
            ]

            if not candidates:
                raise ValueError(f"No vertices of target type {target_type}.")

            non_neighbors = [w for w in candidates if not self.G.has_edge(x, w)]

            if not non_neighbors:
                raise ValueError(
                    f"Vertex {x} of type {x_type} is already connected to all vertices "
                    f"of target type {target_type}."
                )

            y = random.choice(non_neighbors)

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
    builder.transfer_edges_from_cut(num_edges=num_edges_to_transfer, use_ABC=use_ABC, force_ABC_connect_to_ABC=False)

    G = builder.percolation_graph()
    answer, graph = G.is_percolating(return_final_graph=True)
    for v, data in graph.nodes(data=True):
        print(f'vertex {v} of type {data['type']} has degree {graph.degree(v)}.')
    print(answer)

    print(f'Is subgraph of ultra graph? {check_graph_is_subgraph_of_ultra_graph(builder.graph(), s= 2 * number_of_vertices_all_rules + 2, seed=42)}')


    G = builder.percolation_graph()

    answer, final_graph = G.is_percolating(print_steps=False, return_final_graph=True)
    # PercolationGraph(build_dependencies_graph(final_graph)).print_graph()
    if answer:
        nx.write_edgelist(final_graph, 'Counter example!!!')
        raise AssertionError("Final graph is percolating, but it should not be.")

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

    # PercolationGraph(G).print_graph()

def build_dependencies_graph(G : nx.Graph):
    H = nx.Graph()
    for v, data in G.nodes(data=True):
        for u in G.neighbors(v):
            neighbor_type = G.nodes[u].get("type")
            H.add_edge(data['type'], neighbor_type)

    return H


import networkx as nx
import itertools


def build_ultra_graph(s: int, special_per_type: int = 0) -> nx.Graph:
    # K_{s,s,s}
    G = nx.complete_multipartite_graph(s, s, s)

    AB = list(range(0, s))
    BC = list(range(s, 2 * s))
    AC = list(range(2 * s, 3 * s))

    parts = {
        "AB": AB,
        "BC": BC,
        "AC": AC,
    }

    nx.set_node_attributes(G, {v: "AB" for v in AB}, "type")
    nx.set_node_attributes(G, {v: "BC" for v in BC}, "type")
    nx.set_node_attributes(G, {v: "AC" for v in AC}, "type")


    # Independent set ABC
    ABC = list(range(3 * s, 4 * s))
    G.add_nodes_from(ABC)
    nx.set_node_attributes(G, {v: "ABC" for v in ABC}, "type")

    nx.set_node_attributes(G, {v: i for i, v in enumerate(AB)}, "index")
    nx.set_node_attributes(G, {v: i for i, v in enumerate(BC)}, "index")
    nx.set_node_attributes(G, {v: i for i, v in enumerate(AC)}, "index")
    nx.set_node_attributes(G, {v: i for i, v in enumerate(ABC)}, "index")

    # Connections ABC -> one in each part
    G.add_edges_from(zip(ABC, AB))
    G.add_edges_from(zip(ABC, BC))
    G.add_edges_from(zip(ABC, AC))

    parts["ABC"] = ABC

    # Special vertices:
    # each has degree 3, one edge to ABC and one edge to two of A,B,C.
    next_node = 4 * s

    for X, Y in itertools.combinations(["AB", "BC", "AC"], 2):
        special_type = f"ABC-{X}-{Y}"

        for i in range(special_per_type):
            v = next_node
            next_node += 1

            G.add_node(v, type=special_type, index=i % s)

            G.add_edge(v, ABC[i % s])
            G.add_edge(v, parts[X][i % s])
            G.add_edge(v, parts[Y][i % s])

    return G

def build_ultra_graph_randomized(
    s: int,
    special_per_type: int = 0,
    seed=None,
) -> nx.Graph:
    rng = random.Random(seed)

    # Base K_{s,s,s}
    G = nx.complete_multipartite_graph(s, s, s)

    AB = list(range(0, s))
    BC = list(range(s, 2 * s))
    AC = list(range(2 * s, 3 * s))

    parts = {
        "AB": AB,
        "BC": BC,
        "AC": AC,
    }

    nx.set_node_attributes(G, {v: "AB" for v in AB}, "type")
    nx.set_node_attributes(G, {v: "BC" for v in BC}, "type")
    nx.set_node_attributes(G, {v: "AC" for v in AC}, "type")

    # Independent set ABC
    ABC = list(range(3 * s, 4 * s))
    G.add_nodes_from(ABC)
    nx.set_node_attributes(G, {v: "ABC" for v in ABC}, "type")

    # Random ABC connections:
    # each ABC vertex connects to one vertex from AB, one from BC, one from AC
    for v in ABC:
        for part_name in ["AB", "BC", "AC"]:
            u = rng.choice(parts[part_name])
            G.add_edge(v, u)

    parts["ABC"] = ABC

    # Special vertices:
    # each connects to:
    # - one random ABC vertex
    # - one random vertex from each of two chosen base parts
    next_node = 4 * s

    for X, Y in itertools.combinations(["AB", "BC", "AC"], 2):
        special_type = f"ABC-{X}-{Y}"

        for _ in range(special_per_type):
            v = next_node
            next_node += 1

            G.add_node(v, type=special_type)

            abc_neighbor = rng.choice(ABC)
            x_neighbor = rng.choice(parts[X])
            y_neighbor = rng.choice(parts[Y])

            G.add_edge(v, abc_neighbor)
            G.add_edge(v, x_neighbor)
            G.add_edge(v, y_neighbor)

    return G


def check_graph_is_subgraph_of_ultra_graph(G: nx.Graph, s: int, seed=None) -> bool:
    U = build_ultra_graph(s)

    if G.number_of_nodes() > U.number_of_nodes():
        return False
    if G.number_of_edges() > U.number_of_edges():
        return False

    GM = isomorphism.GraphMatcher(U, G)
    return GM.subgraph_is_monomorphic()


if __name__ == "__main__":
    for s in range(2, 8):
        for special_per_type in range(0, 5):
            vals = []
            all_special_degrees = []

            for seed in range(100):
                U = build_ultra_graph_randomized(
                    s=s,
                    special_per_type=special_per_type,
                    seed=seed,
                )

                answer, F = Graph(U).is_percolating(return_final_graph=True)
                vals.append((answer, F.number_of_edges()))

                special_degrees = [
                    F.degree(v)
                    for v, data in F.nodes(data=True)
                    if data["type"].startswith("ABC-")
                ]
                all_special_degrees.extend(special_degrees)

            print(
                "s=", s,
                "special_per_type=", special_per_type,
                "percolated=", sum(a for a, _ in vals),
                "min_edges=", min(e for _, e in vals),
                "max_edges=", max(e for _, e in vals),
                "max_special_degree=", max(all_special_degrees, default=0),
                "special_degrees=", sorted(set(all_special_degrees)),
            )


