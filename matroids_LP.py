import logging
import math
import multiprocessing
import warnings
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from multiprocessing import Pool

import pulp
from tqdm import tqdm

from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def edge_list_kn(n: int):
    return list(itertools.combinations(range(n), 2))


def edge_index_dict(edges):
    return {e: i for i, e in enumerate(edges)}


def graph_from_mask(n: int, edges, mask: int):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    i = 0
    while mask != 0:
        if mask & 1 == 1:
            G.add_edge(*edges[i])
        mask >>= 1
        i += 1

    return G


def graph_to_mask(G, edge_to_index):
    mask = 0

    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        mask |= 1 << edge_to_index[e]

    return mask


class ClassCache:
    @staticmethod
    def _build_iso_lookup(reps):
        buckets = defaultdict(list)

        for i, G in enumerate(reps):
            h = nx.weisfeiler_lehman_graph_hash(G)
            buckets[h].append((i, G))

        return buckets

    def __init__(self, n, reps, kn_edges, edge_to_index):
        super().__init__()
        self.n = n
        self.iso_buckets = self._build_iso_lookup(reps)
        self.kn_edges = kn_edges
        self.edge_to_index = edge_to_index
        self.cache = OrderedDict()

    def _class_id(self, G):
        h = nx.weisfeiler_lehman_graph_hash(G)

        for i, H in self.iso_buckets[h]:
            if nx.is_isomorphic(G, H):
                return i

        raise ValueError("Graph class not found.")

    def preprocess(self, keys):
        for key in tqdm(keys):
            self.cache[key] = self[key]

    def __getitem__(self, item):
        if item not in self.cache:
            self.cache[item] = self._class_id(graph_from_mask(self.n, self.kn_edges, item))

        return self.cache[item]


class K222MatroidProblem:
    def __init__(self, n: int):
        m = n * (n - 1) // 2
        logger.info(f"Initializing base problem with |V|={n}, |E(K_n)|={m}")
        reps = load_unlabeled_graphs_of_order(n)
        kn_edges = edge_list_kn(n)
        edge_to_index = edge_index_dict(kn_edges)

        self.n = n
        self.m = m
        self.reps = reps
        self.kn_edges = kn_edges
        self.edge_to_index = edge_to_index
        self.class_cache = ClassCache(n, reps, kn_edges, edge_to_index)
        self.prob = pulp.LpProblem("symmetric_polymatroid_rank", pulp.LpMaximize)
        self.r = {
            i: pulp.LpVariable(f"r_{i}", lowBound=0, cat=pulp.LpInteger)
            for i in range(len(reps))
        }

    def preprocess_class_cache(self, workers):
        total = 1 << self.m
        chunk_size = (total + workers - 1) // workers

        logger.info(f"Precomputing {total} masks")
        logger.info(f"Workers: {workers}")
        logger.info(f"Chunk size: {chunk_size}")

        tasks = []

        for worker_id in range(workers):
            start = worker_id * chunk_size
            end = min(total, start + chunk_size)

            if start < end:
                tasks.append(range(start, end))

        with Pool(processes=workers) as pool:
            pool.map(self.class_cache.preprocess, tasks)

        logger.info(f"Finished preprocessing")

    def _add_size_constraints(self):
        for i, G in enumerate(self.reps):
            self.prob += self.r[i] <= G.number_of_edges()

    @staticmethod
    def _monotonicity_and_submodularity_worker(args):
        worker, prob, start, end = args

        monotonicity_constraints = set()
        submodularity_constraints = set()

        log_interval = (end - start) // 100
        for g in range(start, end):
            if (g - start) % log_interval == 0:
                logger.info(f"{worker}: {(g - start) / (end - start):3f}%")

            g_id = prob.class_cache[g]
            missing_edges = [1 << e for e in range(prob.m) if not (g & (1 << e))]
            for (i, e_mask) in enumerate(missing_edges):
                # Monotonicity
                ge_id = prob.class_cache[g | e_mask]
                monotonicity_constraints.add((g_id, ge_id))

                # Submodularity
                for f_mask in missing_edges[i:]:
                    gf = g | f_mask
                    gef = gf | e_mask

                    gf_id = prob.class_cache[gf]
                    gef_id = prob.class_cache[gef]

                    left_1, left_2 = sorted((ge_id, gf_id))
                    submodularity_constraints.add((left_1, left_2, g_id, gef_id))

        return monotonicity_constraints, submodularity_constraints

    def _add_monotonicity_and_submodularity_constraints(self, workers: int):
        total = 1 << self.m
        chunk_size = int(math.ceil(total / workers))

        logger.info(f"Generating monotonicity and submodularity constraints using {workers} workers...")
        logger.info(f"Chunk size: {chunk_size}")

        tasks = []

        for worker_id in range(workers):
            start = worker_id * chunk_size
            end = min(total, start + chunk_size)

            if start < end:
                tasks.append((worker_id, self, start, end))

        with Pool(processes=workers) as pool:
            results = pool.map(self._monotonicity_and_submodularity_worker, tasks)

        monotonicity_constraints = set()
        submodularity_constraints = set()
        for work_mono_const, work_sub_const in results:
            monotonicity_constraints.update(work_mono_const)
            submodularity_constraints.update(work_sub_const)

        r = self.r
        for id_A, id_B in monotonicity_constraints:
            self.prob += r[id_A] <= r[id_B]

        for id_Ae, id_Af, id_A, id_Aef in submodularity_constraints:
            self.prob += r[id_Ae] + r[id_Af] >= r[id_A] + r[id_Aef]

    @staticmethod
    def _complete_graph_mask_on_subset(subset, edge_to_index):
        mask = 0

        for u, v in itertools.combinations(subset, 2):
            e = tuple(sorted((u, v)))
            mask |= 1 << edge_to_index[e]

        return mask

    def _add_k5_constraints(self):
        k5_mask = self._complete_graph_mask_on_subset(tuple(range(5)), self.edge_to_index)
        k5_id = self.class_cache[k5_mask]

        # K5 has 10 edges. Circuit means rank exactly 9.
        self.prob += self.r[k5_id] == 9

        for e_idx in range(self.m):
            if (k5_mask >> e_idx) & 1:
                k5_minus_e = k5_mask ^ (1 << e_idx)
                self.prob += self.r[self.class_cache[k5_minus_e]] == 9

    @staticmethod
    def _k222_mask_on_partition(A, B, C, edge_to_index):
        mask = 0

        for X, Y in [(A, B), (A, C), (B, C)]:
            for x in X:
                for y in Y:
                    e = tuple(sorted((x, y)))
                    mask |= 1 << edge_to_index[e]

        return mask

    def _add_k222_constraints(self):
        k222_mask = self._k222_mask_on_partition(A=(0, 1), B=(2, 3), C=(4, 5), edge_to_index=self.edge_to_index)
        k222_id = self.class_cache[k222_mask]

        # K222 has 12 edges. Circuit means rank exactly 11.
        self.prob += self.r[k222_id] == 11

        for e_idx in range(self.m):
            if (k222_mask >> e_idx) & 1:
                k222_minus_e = k222_mask ^ (1 << e_idx)
                self.prob += self.r[self.class_cache[k222_minus_e]] == 11

        full_mask = (1 << self.m) - 1
        full_id = self.class_cache[full_mask]
        self.prob += self.r[full_id]

    def build(self, workers):
        logger.info("Building problem")

        logger.info("Adding empty graph constraint...")
        empty_mask = 0
        empty_id = self.class_cache[empty_mask]
        self.prob += self.r[empty_id] == 0

        logger.info("Adding size constraints...")
        self._add_size_constraints()

        logger.info("Adding monotonicity and elementary submodularity constraints")
        self._add_monotonicity_and_submodularity_constraints(workers)

        logger.info("Adding K5 circuit constraints...")
        self._add_k5_constraints()

        logger.info("Adding K222 circuit constraints...")
        self._add_k222_constraints()

        logger.info("Adding full graph constraints...")
        full_mask = (1 << self.m) - 1
        full_id = self.class_cache[full_mask]
        self.prob += self.r[full_id]

    def load(self):
        pass

    def save(self):
        pass

    def solve(self):
        solver = pulp.HiGHS(msg=False)

        logger.info("Solving...")
        self.prob.solve(solver)

        logger.debug(f"Status: {pulp.LpStatus[self.prob.status]}")

        full_mask = (1 << self.m) - 1
        full_id = self.class_cache[full_mask]
        return pulp.value(self.r[full_id])


def solve_with_all_elementary_submodularity(args):
    prob = K222MatroidProblem(args.n)

    if args.preprocess:
        logger.info("Preprocessing class cache")
        prob.preprocess_class_cache(args.workers)

    prob.build(args.workers)
    max_kn_rank = prob.solve()
    logger.info(f"Solved ILP problem on {args.n} vertices: {max_kn_rank}")
    prob.save()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, help="Number of vertices")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of multiprocess workers")
    parser.add_argument("--preprocess", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    solve_with_all_elementary_submodularity(args)


if __name__ == "__main__":
    main()
