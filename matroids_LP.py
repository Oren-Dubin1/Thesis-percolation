import logging
import multiprocessing
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from multiprocessing import Pool

import pulp

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
        for key in keys:
            self[key] = self[key]

    def __getitem__(self, item):
        if item not in self.cache:
            self.cache[item] = self._class_id(graph_from_mask(self.n, self.kn_edges, item))

        return self.cache[item]

    def __setitem__(self, key, value):
        self.cache[key] = value


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

        logger.info(f"Precomputing {total:,} masks")
        logger.info(f"Workers: {workers}")
        logger.info(f"Chunk size: {chunk_size:,}")

        tasks = []

        for worker_id in range(workers):
            start = worker_id * chunk_size
            end = min(total, start + chunk_size)

            if start < end:
                tasks.append((worker_id, self, start, end))

        class_cache = [None] * total

        t0 = time.time()

        with Pool(processes=workers) as pool:
            pool.map(self._elementary_submodularity_worker, tasks)

        logger.info(f"Finished precomputation in {time.time() - t0:.1f}s")

        return class_cache

    def _add_size_constraints(self):
        logger.info("Adding size constraints")
        for i, G in enumerate(self.reps):
            self.prob += self.r[i] <= G.number_of_edges()

    @staticmethod
    def _monotonicity_worker(args):
        worker, prob, start, end = args
        constraints_batch = set()
        log_step = (end - start) // 10
        for mask in range(start, end):
            if (mask - start) % log_step == 0:
                logger.info(f"Worker {worker}: {mask}/{end}")

            id_a = prob.class_cache[mask]

            for e_idx in range(prob.m):
                if not ((mask >> e_idx) & 1):
                    id_b = prob.class_cache[mask | (1 << e_idx)]
                    constraints_batch.add((id_a, id_b))

        return constraints_batch

    def _add_monotonicity_constraints(self, workers: int):
        total = 1 << self.m
        chunk_size = (total + workers - 1) // workers

        tasks = []
        for worker_id in range(workers):
            start_mask = worker_id * chunk_size
            end_mask = min(total, start_mask + chunk_size)

            if start_mask < end_mask:
                tasks.append((worker_id, self, start_mask, end_mask))

        logger.info(f"Generating monotonicity constraints using {workers} workers...")
        with Pool(processes=workers) as pool:
            results = pool.map(self._monotonicity_worker, tasks)

        constraints = set()

        for worker_constraints in results:
            constraints.update(worker_constraints)

        logger.info(f"Distinct monotonicity constraints: {len(constraints)}")
        logger.info("Adding monotonicity constraints to PuLP model...")

        for id_A, id_B in constraints:
            self.prob += self.r[id_A] <= self.r[id_B]

    @staticmethod
    def _elementary_submodularity_worker(args):
        worker_id, prob, start_A, end_A = args

        constraints = set()

        for A in range(start_A, end_A):
            if (A - start_A) % 100_000 == 0:
                logger.info(f"Worker {worker_id}: processed A={A}/{end_A}")

            class_cache = prob.class_cache
            id_A = class_cache[A]
            missing = [e for e in range(prob.m) if not ((A >> e) & 1)]

            for i in range(len(missing)):
                e = missing[i]
                Ae = A | (1 << e)
                id_Ae = class_cache[Ae]

                for j in range(i + 1, len(missing)):
                    f = missing[j]
                    Af = A | (1 << f)
                    Aef = Ae | (1 << f)

                    id_Af = class_cache[Af]
                    id_Aef = class_cache[Aef]

                    left_1, left_2 = sorted((id_Ae, id_Af))
                    constraints.add((left_1, left_2, id_A, id_Aef))

        return constraints

    def _add_elementary_submodularity_constraints(self, workers: int):
        logger.info(f"Generating elementary submodularity constraints using {workers} workers...")

        total = 1 << self.m
        chunk_size = (total + workers - 1) // workers

        tasks = []

        for worker_id in range(workers):
            start_A = worker_id * chunk_size
            end_A = min(total, start_A + chunk_size)

            if start_A < end_A:
                tasks.append((worker_id, self, start_A, end_A))

        with Pool(processes=workers) as pool:
            results = pool.map(self._elementary_submodularity_worker, tasks)

        constraints = set()

        for worker_constraints in results:
            constraints.update(worker_constraints)

        logger.info(f"Distinct elementary submodularity constraints: {len(constraints)}")
        logger.info("Adding constraints to PuLP model...")

        r = self.r
        for id_Ae, id_Af, id_A, id_Aef in constraints:
            self.prob += r[id_Ae] + r[id_Af] >= r[id_A] + r[id_Aef]

        return len(constraints)

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

        logger.info("Preprocessing class cache")
        self.preprocess_class_cache(workers)

        logger.info("Adding size constraints...")
        self._add_size_constraints()

        logger.info("Adding monotonicity constraints...")
        self._add_monotonicity_constraints(workers)

        logger.info("Adding elementary submodularity constraints...")
        self._add_elementary_submodularity_constraints(workers)

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

    prob.build(args.workers)
    max_kn_rank = prob.solve()
    logger.info(f"Solved ILP problem on {args.n} vertices: {max_kn_rank}")
    prob.save()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, help="Number of vertices")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of multiprocess workers")

    return parser.parse_args()


def main():
    args = parse_args()
    solve_with_all_elementary_submodularity(args)


if __name__ == "__main__":
    main()
