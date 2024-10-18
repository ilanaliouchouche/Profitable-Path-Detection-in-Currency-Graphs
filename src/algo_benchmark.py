import numpy as np
import pandas as pd
import random
from time import perf_counter
from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from typing import Sequence, Callable, Optional, Tuple, List
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import tracemalloc


class AlgoBenchmark:
    """
    The goal of this class is to compare the performance (time and complexity)
    of the different algorithms implemented in `src/optimal_paths.py`.
    """

    def __init__(self,
                 seed: int = 42) -> None:
        """
        Initializes the benchmarking class.

        ## Parameters
            `seed`: Seed for the random number generator.

        ## Example
        ```py

        from src.algo_benchmark import AlgoBenchmark
        benchmark = AlgoBenchmark(seed=42)
        ```
        """

        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.time_results = defaultdict(lambda: defaultdict(list))
        self.complexity_results = defaultdict(lambda: defaultdict(list))
        self.memory_results = defaultdict(lambda: defaultdict(list))
        self.__ran = False

    def run(self,
            algorithms: Sequence[Callable[
                [CurrencyGraph, CurrencyNode, Optional[int],
                 Optional[Callable[[CurrencyNode], None]],
                 Optional[Callable[[CurrencyEdge], None]]],
                Tuple[List[CurrencyNode], float]]],
            node_sizes: Sequence[int],
            num_trials: int = 10,
            scale: float = 0.4,
            n_passages_fn: Optional[Callable[[CurrencyGraph], int]] = None,
            verbose: bool = False
            ) -> Tuple[defaultdict, defaultdict, defaultdict]:
        """
        Runs the provided algorithms and measures both execution time,
        memory usage, and complexity.

        ## Parameters
            `algorithms`: List of algorithms to benchmark.
            `node_sizes`: List of number of nodes to simulate.
            `num_trials`: Number of trials to run for each number of nodes.
            `scale`: Scale parameter for the normal distribution of edge
                     weights.
            `n_passages_fn`: Function to calculate the number of passages.
            `verbose`: Whether to print the results.

        ## Returns
            A tuple containing time, complexity, and memory usage statistics.

        ## Example
        ```py
        from src.algo_benchmark import AlgoBenchmark
        from src.optimal_paths import simplified_dijkstra, brute_force

        benchmark = AlgoBenchmark(seed=42)
        time_results, complexity_results, memory_results = benchmark.run(
            algorithms=[simplified_dijkstra, brute_force],
            node_sizes=[5, 10, 15],
            num_trials=10,
            scale=0.4
        )
        ```
        """

        if self.__ran:
            raise ValueError("This benchmark has already been run. "
                             "Please create a new instance to run another "
                             "benchmark.")

        self._ran = True

        np.random.seed(self.seed)
        random.seed(self.seed)
        results = defaultdict(lambda: defaultdict(list))
        complexity_results = defaultdict(lambda: defaultdict(list))
        memory_results = defaultdict(lambda: defaultdict(list))

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
                    visited_nodes = set()
                    traversed_edges = 0

                    def node_callback(node):
                        visited_nodes.add(node)

                    def edge_callback(edge):
                        nonlocal traversed_edges
                        traversed_edges += 1

                    tracemalloc.start()

                    start_time = perf_counter()

                    try:
                        algo(G, nodes[0], n_passages,
                             node_callback=node_callback,
                             edge_callback=edge_callback)
                    except TypeError:
                        algo(G, nodes[0], node_callback=node_callback,
                             edge_callback=edge_callback)

                    elapsed_time = perf_counter() - start_time

                    current_usage, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    memory_used = peak / (1024 * 1024)

                    results[algo.__name__][num_nodes].append(elapsed_time)
                    memory_results[algo.__name__][num_nodes].append(
                        memory_used)

                    complexity_results[algo.__name__][num_nodes].append({
                        'nodes': len(visited_nodes),
                        'edges': traversed_edges
                    })

        for algo, timings in results.items():
            if verbose:
                print(f"\n{algo} Results:")
            for num_nodes, times in timings.items():
                avg_time = np.mean(times)
                ci_lower = np.percentile(times, 2.5)
                ci_upper = np.percentile(times, 97.5)

                avg_memory = np.mean(memory_results[algo][num_nodes])
                ci_lower_memory = np.percentile(memory_results[algo]
                                                [num_nodes], 2.5)
                ci_upper_memory = np.percentile(memory_results[algo]
                                                [num_nodes], 97.5)

                avg_nodes = np.mean([comp['nodes'] for comp in
                                    complexity_results[algo][num_nodes]])
                ci_nodes_low = np.percentile([comp['nodes'] for comp in
                                              complexity_results[algo]
                                              [num_nodes]], 2.5)
                ci_nodes_up = np.percentile([comp['nodes'] for comp in
                                            complexity_results[algo]
                                            [num_nodes]], 97.5)

                avg_edges = np.mean([comp['edges'] for comp in
                                    complexity_results[algo][num_nodes]])
                ci_edges_low = np.percentile([comp['edges'] for comp in
                                              complexity_results[algo]
                                              [num_nodes]], 2.5)
                ci_edges_up = np.percentile([comp['edges'] for comp in
                                            complexity_results[algo]
                                            [num_nodes]], 97.5)
                if verbose:
                    print(f"  {algo} Avg Time: {avg_time:.6f}s (95% CI: "
                          f"[{ci_lower:.6f}, {ci_upper:.6f}])")
                    print(f"  {algo} Avg Memory Usage: {avg_memory:.2f} MB "
                          f"(95% CI: [{ci_lower_memory:.2f}, "
                          f"{ci_upper_memory:.2f}])")
                    print(f"  {algo} Avg Nodes Visited: {avg_nodes:.2f} "
                          f"(95% CI: [{ci_nodes_low:.2f}, {ci_nodes_up:.2f}])")
                    print(f"  {algo} Avg Edges Traversed: {avg_edges:.2f} "
                          f"(95% CI: [{ci_edges_low:.2f}, {ci_edges_up:.2f}])")

        self.time_results = results
        self.complexity_results = complexity_results
        self.memory_results = memory_results

        return results, complexity_results, memory_results

    @staticmethod
    def plot_time_with_ci(results: defaultdict) -> None:
        """
        Static method to plot execution time vs number of samples,
        with confidence intervals.

        ## Parameters
            `results`: Dictionary containing the execution times and
                       confidence intervals.

        ## Example
        ```py
        from src.optimal_paths import simplified_dijkstra, brute_force

        benchmark = AlgoBenchmark(seed=42)
        results, _ = benchmark.run(
            algorithms=[simplified_dijkstra, brute_force],
            node_sizes=[5, 10, 15],
            num_trials=10,
            scale=0.4
        )
        AlgoBenchmark.plot_time_with_ci(results)
        ```
        """

        plt.style.use('ggplot')

        plt.figure(figsize=(10, 6))

        for algo, timings in results.items():
            node_sizes = sorted(timings.keys())
            avg_times = [np.mean(timings[n]) for n in node_sizes]
            ci_lowers = [np.percentile(timings[n], 2.5) for n in node_sizes]
            ci_uppers = [np.percentile(timings[n], 97.5) for n in node_sizes]

            plt.plot(node_sizes, avg_times, label=algo, marker='o')

            plt.fill_between(node_sizes, ci_lowers, ci_uppers, alpha=0.2)

        plt.xlabel("Number of Nodes (N)")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Algorithm Performance: Execution Time vs Number of Nodes")
        plt.legend()

        plt.show()

    @staticmethod
    def plot_complexity_with_ci(complexity_results: dict) -> None:
        """
        Static method to plot complexity vs number of samples, with confidence
        intervals. Plots two graphs: one for nodes visited and one for edges.

        ## Parameters
            `complexity_results`: Dictionary containing the complexity results.

        ## Example
        ```py
        from src.optimal_paths import simplified_dijkstra, brute_force

        benchmark = AlgoBenchmark(seed=42)
        _, complexity_results = benchmark.run(
            algorithms=[simplified_dijkstra, brute_force],
            node_sizes=[5, 10, 15],
            num_trials=10,
            scale=0.4
        )
        AlgoBenchmark.plot_complexity_with_ci(complexity_results)
        ```
        """

        plt.style.use('ggplot')

        _, (node_ax, edge_ax) = plt.subplots(1, 2, figsize=(14, 6))

        for algo, data in complexity_results.items():
            node_sizes = sorted(data.keys())
            avg_nodes = [np.mean(
                [comp['nodes'] for comp in data[n]]) for n in node_sizes]
            ci_nodes_lowers = [np.percentile(
                [comp['nodes'] for comp in data[n]], 2.5) for n in node_sizes]
            ci_nodes_uppers = [np.percentile(
                [comp['nodes'] for comp in data[n]], 97.5) for n in node_sizes]

            node_ax.plot(node_sizes, avg_nodes, label=algo, marker='o')
            node_ax.fill_between(node_sizes, ci_nodes_lowers, ci_nodes_uppers,
                                 alpha=0.2)

        node_ax.set_xlabel("Number of Nodes (N)")
        node_ax.set_ylabel("Nodes Visited")
        node_ax.set_title("Algorithm Complexity: Nodes Visited vs Number of "
                          "Nodes")
        node_ax.legend()

        for algo, data in complexity_results.items():
            node_sizes = sorted(data.keys())
            avg_edges = [np.mean(
                [comp['edges'] for comp in data[n]]) for n in node_sizes]
            ci_edges_lowers = [np.percentile(
                [comp['edges'] for comp in data[n]], 2.5) for n in node_sizes]
            ci_edges_uppers = [np.percentile(
                [comp['edges'] for comp in data[n]], 97.5) for n in node_sizes]

            edge_ax.plot(node_sizes, avg_edges, label=algo, marker='o')
            edge_ax.fill_between(node_sizes, ci_edges_lowers, ci_edges_uppers,
                                 alpha=0.2)

        edge_ax.set_xlabel("Number of Nodes (N)")
        edge_ax.set_ylabel("Edges Traversed")
        edge_ax.set_title("Algorithm Complexity: Edges Traversed vs Number of "
                          "Nodes")
        edge_ax.legend()

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_memory_with_ci(memory_results: defaultdict) -> None:
        """
        Static method to plot memory usage vs number of nodes,
        with confidence intervals.

        ## Parameters
            `memory_results`: Dictionary containing memory usage results.

        ## Example
        ```py
        from src.optimal_paths import simplified_dijkstra, brute_force

        benchmark = AlgoBenchmark(seed=42)
        _, _, memory_results = benchmark.run(
            algorithms=[simplified_dijkstra, brute_force],
            node_sizes=[5, 10, 15],
            num_trials=10,
            scale=0.4
        )
        AlgoBenchmark.plot_memory_with_ci(memory_results)
        ```
        """

        plt.style.use('ggplot')

        plt.figure(figsize=(10, 6))

        for algo, data in memory_results.items():
            node_sizes = sorted(data.keys())
            avg_memory = [np.mean(data[n]) for n in node_sizes]
            ci_memory_lower = [np.percentile(data[n], 2.5)
                               for n in node_sizes]
            ci_memory_upper = [np.percentile(data[n], 97.5)
                               for n in node_sizes]

            plt.plot(node_sizes, avg_memory, label=algo, marker='o')

            plt.fill_between(node_sizes, ci_memory_lower, ci_memory_upper,
                             alpha=0.2)

        plt.xlabel("Number of Nodes (N)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Algorithm Performance: Memory Usage vs Number of Nodes")
        plt.legend()

        plt.show()

    def to_csv(self,
               output_file: str) -> None:
        """
        Converts the time and complexity results into a DataFrame and saves
        them as a CSV file.

        ## Parameters
            `output_file`: The path where the CSV file will be saved.

        ## Example
        ```py
        from src.optimal_paths import simplified_dijkstra, brute_force

        benchmark = AlgoBenchmark(seed=42)
        _, _ = benchmark.run(
            algorithms=[simplified_dijkstra, brute_force],
            node_sizes=[5, 10, 15],
            num_trials=10,
            scale=0.4
        )
        benchmark.to_csv("benchmark_results.csv")
        ```
        """

        data = []

        for algo, node_sizes in self.time_results.items():
            for num_nodes, times in node_sizes.items():
                avg_time = pd.Series(times).mean()
                ci_lower_time = pd.Series(times).quantile(0.025)
                ci_upper_time = pd.Series(times).quantile(0.975)

                avg_nodes = pd.Series(
                    [comp['nodes'] for comp in self.complexity_results[algo]
                     [num_nodes]]).mean()
                ci_lower_nodes = pd.Series(
                    [comp['nodes'] for comp in self.complexity_results[algo]
                     [num_nodes]]).quantile(0.025)
                ci_upper_nodes = pd.Series(
                    [comp['nodes'] for comp in self.complexity_results[algo]
                     [num_nodes]]).quantile(0.975)

                avg_edges = pd.Series(
                    [comp['edges'] for comp in self.complexity_results[algo]
                     [num_nodes]]).mean()
                ci_lower_edges = pd.Series(
                    [comp['edges'] for comp in self.complexity_results[algo]
                     [num_nodes]]).quantile(0.025)
                ci_upper_edges = pd.Series(
                    [comp['edges'] for comp in self.complexity_results[algo]
                     [num_nodes]]).quantile(0.975)

                avg_memory = pd.Series(self.memory_results[algo]
                                       [num_nodes]).mean()
                ci_lower_memory = pd.Series(self.memory_results[algo]
                                            [num_nodes]).quantile(0.025)
                ci_upper_memory = pd.Series(self.memory_results[algo]
                                            [num_nodes]).quantile(0.975)

                data.append({
                    'Algorithm': algo,
                    'Number of Nodes': num_nodes,
                    'Avg Time (s)': avg_time,
                    'CI Lower Time (s)': ci_lower_time,
                    'CI Upper Time (s)': ci_upper_time,
                    'Avg Nodes Visited': avg_nodes,
                    'CI Lower Nodes': ci_lower_nodes,
                    'CI Upper Nodes': ci_upper_nodes,
                    'Avg Edges Traversed': avg_edges,
                    'CI Lower Edges': ci_lower_edges,
                    'CI Upper Edges': ci_upper_edges,
                    'Avg Memory Usage (MB)': avg_memory,
                    'CI Lower Memory (MB)': ci_lower_memory,
                    'CI Upper Memory (MB)': ci_upper_memory
                })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        return df


if __name__ == "__main__":

    from src.optimal_paths import (simplified_dijkstra,
                                   brute_force,
                                   log_brute_force)
    from datetime import datetime
    import os

    BENCH_APP = os.path.join(os.path.dirname(__file__), '../app/bench_app/')

    today = datetime.today().strftime('%Y-%m-%d')

    benchmark = AlgoBenchmark(seed=42)
    time_results, complexity_results, memory_results = benchmark.run(
        algorithms=[simplified_dijkstra, brute_force, log_brute_force],
        node_sizes=[2, 3, 4, 5, 6, 7],
        num_trials=10,
        scale=0.4
    )

    # benchmark.plot_time_with_ci(time_results)
    # benchmark.plot_complexity_with_ci(complexity_results)
    # benchmark.plot_memory_with_ci(memory_results)

    benchmark.to_csv(f"{BENCH_APP}results.csv")
