from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections import deque
from tabnanny import check
from typing import Dict, Iterable, List, Optional, Set, Tuple
from Graphs import PercolationGraph

import networkx as nx

Edge = Tuple[int, int]


def norm_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class Witness:
    rule: str  # "K5" or "K222"
    vertices: Tuple[int, ...]
    parts: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None  # for K222 only


@dataclass
class Event:
    step_idx: int
    added_edge: Edge
    witness: Witness
    could_add_without_v: bool


def remove_vertex_relabel(G: nx.Graph, v: int) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Remove vertex v and relabel remaining vertices to 1..n-1 preserving order.
    Returns (G_minus_v, old_to_new_map).
    """
    nodes = sorted(G.nodes())
    if v not in nodes:
        raise ValueError(f"vertex {v} not in G")
    keep = [u for u in nodes if u != v]
    mp = {u: (i + 1) for i, u in enumerate(keep)}  # 1..n-1
    H = nx.Graph()
    H.add_nodes_from(mp.values())
    for a, b in G.edges():
        if a == v or b == v:
            continue
        H.add_edge(mp[a], mp[b])
    return H, mp


def all_pair_partitions_6(V6: Tuple[int, ...]) -> Iterable[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    All partitions of 6 vertices into 3 unordered pairs (canonical generation).
    """
    v = list(V6)
    v0 = v[0]
    for i in range(1, 6):
        p1 = tuple(sorted((v0, v[i])))
        rest = [v[j] for j in range(1, 6) if j != i]
        r0 = rest[0]
        for j in range(1, 4):
            p2 = tuple(sorted((r0, rest[j])))
            r = [rest[k] for k in range(1, 4) if k != j]  # two verts
            p3 = tuple(sorted((r[0], r[1])))
            # canonical order of pairs to avoid duplicates from swapping B,C
            pairs = tuple(sorted((p1, p2, p3)))
            yield pairs


def k5_witnesses(G: nx.Graph, e: Edge) -> List[Witness]:
    u, v = e
    if G.has_edge(u, v):
        return []
    out: List[Witness] = []
    others = [z for z in G.nodes() if z not in (u, v)]
    for a, b, c in combinations(others, 3):
        S = (u, v, a, b, c)
        ok = True
        for p, q in combinations(S, 2):
            if norm_edge(p, q) == norm_edge(u, v):
                continue
            if not G.has_edge(p, q):
                ok = False
                break
        if ok:
            out.append(Witness(rule="K5", vertices=tuple(sorted(S))))
    return out


def k222_witnesses(G: nx.Graph, e: Edge) -> List[Witness]:
    """
    Edge e=(s,t) addable by K222 if there exist 4 other vertices and a partition into 3 pairs
    such that s,t are in different parts and all cross-edges exist except (s,t).
    """
    s, t = e
    if G.has_edge(s, t):
        return []
    out: List[Witness] = []
    others = [z for z in G.nodes() if z not in (s, t)]
    for a, b, c, d in combinations(others, 4):
        V6 = tuple(sorted((s, t, a, b, c, d)))
        for parts in all_pair_partitions_6(V6):
            # part index lookup
            part_of: Dict[int, int] = {}
            for idx, pr in enumerate(parts):
                part_of[pr[0]] = idx
                part_of[pr[1]] = idx
            if part_of[s] == part_of[t]:
                continue  # then st is inside a part, not forced
            ok = True
            for p, q in combinations(V6, 2):
                if part_of[p] != part_of[q]:
                    if norm_edge(p, q) == norm_edge(s, t):
                        continue
                    if not G.has_edge(p, q):
                        ok = False
                        break
            if ok:
                out.append(Witness(rule="K222", vertices=V6, parts=parts))
    return out

def witnesses_for_edge(G: nx.Graph, e: Edge) -> List[Witness]:
    wits: List[Witness] = []
    wits.extend(k5_witnesses(G, e))
    wits.extend(k222_witnesses(G, e))
    return wits


def addable_edges_with_witnesses(G: nx.Graph) -> Dict[Edge, List[Witness]]:
    res: Dict[Edge, List[Witness]] = {}
    nodes = sorted(G.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if G.has_edge(u, v):
                continue
            e = (u, v)
            wits: List[Witness] = []
            wits.extend(k5_witnesses(G, e))
            wits.extend(k222_witnesses(G, e))
            if wits:
                res[e] = wits
    return res


def witness_uses_pair_vx(w: Witness, v: int, x: int) -> bool:
    if w.rule != "K222" or w.parts is None:
        return False
    return any(set(pair) == {v, x} for pair in w.parts)


def edge_addable_without_v_one_step(G: nx.Graph, e: Edge, v: int) -> bool:
    """
    One-step addability in (G - v) (after relabeling). Checks for a K5 or K222 witness in G-v.
    """
    if v in e:
        return False
    H, mp = remove_vertex_relabel(G, v)
    e2 = norm_edge(mp[e[0]], mp[e[1]])
    return bool(k5_witnesses(H, e2) or k222_witnesses(H, e2))


def brute_force_sequences(
    G0: nx.Graph,
    v: int,
    x: int,
    target_vertex: int = 6,
    max_steps: int = 10,
    stop_after: int = 5000,
) -> List[Event]:
    """
    Explore percolation sequences up to max_steps using DFS over states.
    Record events where an edge (i, target_vertex) with i != v,x is added by a K222 witness
    that uses {v,x} as a pair, and check whether that edge is one-step addable in G-v.
    """
    events: List[Event] = []
    seen: Set[frozenset] = set()
    q = deque([(G0.copy(), 0)])
    expansions = 0

    def edge_key(G: nx.Graph) -> frozenset:
        return frozenset(norm_edge(a, b) for a, b in G.edges())

    while q and expansions < stop_after:
        G, step = q.pop()
        key = (edge_key(G), step)
        if key in seen:
            continue
        seen.add(key)

        if step >= max_steps:
            continue

        addable = addable_edges_with_witnesses(G)
        if not addable:
            continue

        for e, wits in addable.items():
            for w in wits:
                G2 = G.copy()
                G2.add_edge(*e)
                expansions += 1

                # Record relevant events: (i,target) edge, not incident to v or x, added via K222 with pair {v,x}
                if w.rule == "K222" and (v not in e) and (x not in e) and (target_vertex in e):
                    other = e[0] if e[1] == target_vertex else e[1]
                    if other not in (v, x) and witness_uses_pair_vx(w, v, x):
                        events.append(
                            Event(
                                step_idx=step + 1,
                                added_edge=norm_edge(*e),
                                witness=w,
                                could_add_without_v=edge_addable_without_v_one_step(G, e, v),
                            )
                        )

                if expansions >= stop_after:
                    break
                q.append((G2, step + 1))
            if expansions >= stop_after:
                break

    return events


# Assumes these already exist in your file:
# - addable_edges_with_witnesses(G) -> Dict[Edge, List[Witness]]
# - Witness dataclass with fields: rule, vertices, parts (parts only for K222)

def percolate_without_v(
    G: nx.Graph,
    v: int,
    *,
    prefer_k5: bool = False,
    return_trace: bool = False,
) -> nx.Graph | Tuple[nx.Graph, List[Tuple[Edge, "Witness"]]]:
    """
    Compute the percolation closure of G under (K5, K222) steps, with the restriction
    that no witness used is allowed to contain vertex v.

    This DOES NOT remove v from the graph; it simply forbids using v inside any witness.
    Edges incident to v may still get added if there exists a witness that does not use v.

    Parameters
    ----------
    G : nx.Graph
        Starting graph.
    v : int
        Forbidden vertex: witnesses containing v are disallowed.
    prefer_k5 : bool
        If True, when multiple addable edges exist, try K5 witnesses first (purely heuristic).
    return_trace : bool
        If True, also returns a list of (added_edge, witness_used) in the order applied.

    Returns
    -------
    nx.Graph
        The closure graph.
    (nx.Graph, trace)
        If return_trace=True.
    """
    H = G.copy()
    trace: List[Tuple[Edge, "Witness"]] = []

    while True:
        addable: Dict[Edge, List["Witness"]] = addable_edges_with_witnesses(H)

        # Filter: keep only witnesses that do NOT use v
        filtered: Dict[Edge, List["Witness"]] = {}
        for e, wits in addable.items():
            w2 = [w for w in wits if v not in w.vertices]
            if w2:
                if prefer_k5:
                    # Deterministic ordering: K5 witnesses first if requested
                    w2.sort(key=lambda w: 0 if w.rule == "K5" else 1)
                filtered[e] = w2

        if not filtered:
            break

        # Apply one step, then recompute (sequential closure)
        # Deterministic choice: smallest edge lexicographically, then first witness
        e = min(filtered.keys())
        w = filtered[e][0]

        if not H.has_edge(*e):
            H.add_edge(*e)
            if return_trace:
                trace.append((e, w))
        else:
            # Shouldn't happen, but avoid infinite loops if input dict includes existing edges
            pass

    return (H, trace) if return_trace else H


def enumerate_graphs_with_added_vertex(G: nx.Graph, v: int=0):
    new_vertex = max(G.nodes()) + 1
    for deg in range(3, G.number_of_nodes()):
        for neighbors in combinations(set(G.nodes())-{v}, deg):
            G2 = G.copy()
            G2.add_node(new_vertex)
            G2.add_edges_from([(new_vertex, nb) for nb in neighbors])
            H = percolate_without_v(G2, v)
            yield H

def check_conj_all_graphs(G: nx.Graph,
                          v: int,
                          x: int,
                          target_vertex: int = 6,
                          depth: int = 1,
                          base_graphs: Optional[Set] = None
                          ) -> bool:
    """
    Check the conjecture for all graphs obtained by adding a new vertex to G
    connected to at least 3 vertices other than v, and percolating without v.
    Recursively check graphs up to given depth.
    :param G: Initial Graph
    :param v:
    :param x:
    :param target_vertex:
    :param depth:
    :param checked_graphs:
    :return:
    """
    print(f"Entering check_conj_all_graphs: depth={depth}, target_vertex={target_vertex}, v={v}, x={x}")

    if base_graphs is None:
        base_graphs = {G}
        print(f"Initialized base_graphs with 1 graph")

    checked_graphs = base_graphs.copy()
    base_list = list(base_graphs)
    print(f"Processing {len(base_list)} base graph(s) at this level")

    for idx, graph in enumerate(base_list, start=1):
        graphs = list(enumerate_graphs_with_added_vertex(graph))
        print(f"    Generated {len(graphs)} graph(s) with an added vertex, depth {depth}")

        for H_idx, H in enumerate(graphs, start=1):
            if H in checked_graphs:
                continue

            witnesses = witnesses_for_edge(H, (target_vertex, v))
            for wit in witnesses:
                if x not in wit.vertices:
                    continue
                print(f"        Counterexample found: witness {wit} includes x={x}; printing graph then raising")
                PercolationGraph(H).print_graph()
                raise ValueError(f"Conjecture failed for graph with added vertex: edge ({target_vertex},{v}) addable by {wit} including x={x}")

            checked_graphs.add(H)

    depth -= 1
    print(f"Level complete. Remaining depth after decrement: {depth}")
    if not depth:
        return True

    print("Recursing to next depth level")
    return check_conj_all_graphs(G, v, x, target_vertex, depth, checked_graphs)





if __name__ == "__main__":
    G = nx.complete_multipartite_graph(2,2,2)
    v = 0
    x = 1
    print(check_conj_all_graphs(G, v, x, target_vertex=5, depth=3))