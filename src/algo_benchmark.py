import numpy as np
import random
from time import perf_counter
from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from typing import Sequence, Callable, Optional, Tuple, List
from collections import defaultdict
from tqdm.auto import tqdm


class AlgoBenchmark:
    """
    The goal of this class is to compare the performance of the different
    algorithms implemented in `src/optimal_paths.py`.
    """

    def __init__(self) -> None:
        self.studies = []

    @staticmethod
    def time_study(algorithms: Sequence[Callable[
                       [CurrencyGraph, CurrencyNode,
                        Optional[int]], Tuple[List[CurrencyNode], float]]],
                   node_sizes: Sequence[int],
                   num_trials: int = 10,
                   scale: float = 0.4,
                   seed: int = 42,
                   n_passages_fn: Optional[Callable[
                       [CurrencyGraph], int]] = None) -> defaultdict:
        """
        Generates random currency graphs for each size in `node_sizes`, runs
        the provided algorithms, and measures the execution time.

        ## Parameters:
            `algorithms`: List of algorithms to compare. Each algorithm should
                          take a graph, a start node, and optionally an
                          integer parameter (n_passages).
            `node_sizes`: List of integers specifying the number of nodes for
                          each graph.
            `num_trials`: Number of trials for each graph size to average
                          the results.
            `scale`: Standard deviation of the normal distribution used to
                     generate edge weights.
            `n_passages_fn`: Optional function that takes a graph and returns
                             an integer (n_passages). If None, n_passages will
                             be set to (number of nodes - 1).
        """

        random.seed(seed)
        results = defaultdict(lambda: defaultdict(list))

        for num_nodes in tqdm(node_sizes, desc="Simulating some graphs"):
            for _ in range(num_trials):
                nodes = [CurrencyNode(chr(65 + i)) for i in range(num_nodes)]
                edges = []
                for i in range(num_nodes):
                    edges.append(CurrencyEdge(nodes[i], nodes[i], 1.0))
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        weight_ij = np.random.normal(loc=1.0, scale=scale)
                        weight_ji = 1 / weight_ij
                        edges.append(CurrencyEdge(nodes[i], nodes[j],
                                                  weight_ij))
                        edges.append(CurrencyEdge(nodes[j], nodes[i],
                                                  weight_ji))
                G = CurrencyGraph(nodes, edges)

                n_passages = (n_passages_fn(G) if n_passages_fn
                              else num_nodes - 1)

                for algo in algorithms:
                    start_time = perf_counter()

                    try:
                        algo(G, nodes[0], n_passages)
                    except TypeError:
                        algo(G, nodes[0])

                    elapsed_time = perf_counter() - start_time
                    results[algo.__name__][num_nodes].append(elapsed_time)

        for num_nodes, timings in results.items():
            print(f"Graph with {num_nodes} nodes:")
            for algo, times in timings.items():
                avg_time = np.mean(times)
                ci_lower = np.percentile(times, 2.5)
                ci_upper = np.percentile(times, 97.5)
                print(f"  {algo} Avg Time: {avg_time:.6f}s "
                      f"(95% CI: [{ci_lower:.6f}, {ci_upper:.6f}])\n")

        return results

    def run(self):
        pass
