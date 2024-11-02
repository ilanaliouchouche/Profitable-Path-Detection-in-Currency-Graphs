from typing import List, Tuple, Any
import pandas as pd
from tqdm.auto import tqdm
from time import perf_counter
import numpy as np
from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from src.optimal_paths import (simplified_dijkstra,
                               log_shifted_simplified_dijkstra)


def create_graph_with_small_weights(n: int, decim: int) -> CurrencyGraph:
    """
    Create a currency graph with small edge weights.

    ## Parameters
        `n`: The number of nodes in the graph.
        `decim`: The decimal exponent for the small edge weights.

    ## Returns
        A currency graph with small and large edge weights.
    """
    nodes = [CurrencyNode(f"{i}") for i in range(n)]
    edges = []

    for i in range(n):
        edges.append(CurrencyEdge(nodes[i], nodes[i], 1.0))

    for i in range(n):
        for j in range(i + 1, n):
            low = 10 ** -decim
            high = 1
            small_weight = np.random.uniform(low, high)
            inverse_weight = 1 / small_weight
            edges.append(CurrencyEdge(nodes[i], nodes[j], small_weight))
            edges.append(CurrencyEdge(nodes[j], nodes[i], inverse_weight))

    edges = [x for i, x in enumerate(edges) if x not in edges[i + 1:]]
    return CurrencyGraph(nodes, edges)


def compare_algorithms_with_small_weights(n: int,
                                          decim: int,
                                          start_node_index: int
                                          ) -> Tuple[Any, Any]:
    """
    Compare simplified Dijkstra and log-shifted Dijkstra on a graph with small
    edge weights and return a tuple of profit values.

    ## Parameters
        `n`: The number of nodes in the graph.
        `decim`: The decimal exponent for the small edge weights.
        `start_node_index`: The index of the node to start the cycle from.

    ## Returns
        Tuple[float, float]: profit from simplified Dijkstra, profit from
                             log-shifted Dijkstra
    """

    G = create_graph_with_small_weights(n, decim)
    start_node = G.nodes[start_node_index]

    try:
        cycle_classic, profit_classic = (
            simplified_dijkstra(G, start_node, len(G.nodes) - 1))
    except Exception:
        profit_classic = -999

    try:
        cycle_log, profit_log = (
            log_shifted_simplified_dijkstra(G, start_node, len(G.nodes) - 1,
                                            return_log_profit=True))
    except Exception:
        profit_log = -999

    return profit_classic, profit_log


def run_overflow_trials(node_sizes: List[int],
                        decimal_sizes: List[int],
                        n_trials: int) -> pd.DataFrame:
    """
    Test simplified Dijkstra and log-shifted Dijkstra algorithms
    for underflow/overflow across multiple node sizes and decimal scales.

    ## Parameters
        `node_sizes`: List of node sizes to test.
        `decimal_sizes`: List of decimal scales to test.
        `n_trials`: Number of trials to run per node size and decimal size.

    ## Returns
        A pandas DataFrame containing the profit values, average execution
        times, and confidence intervals for each node size and decimal size
        for both algorithms.
    """

    results = []

    for n in tqdm(node_sizes, desc="Node Sizes"):
        for decim in tqdm(decimal_sizes, desc="Decimal Sizes"):
            simplified_profits = []
            log_shifted_profits = []

            simplified_times = []
            log_shifted_times = []

            for _ in range(n_trials):
                start_time = perf_counter()
                profit_classic, _ = (
                    compare_algorithms_with_small_weights(n, decim, 0))
                simplified_times.append(perf_counter() - start_time)
                simplified_profits.append(profit_classic)

                start_time = perf_counter()
                _, profit_log_shifted = (
                    compare_algorithms_with_small_weights(n, decim, 0))
                log_shifted_times.append(perf_counter() - start_time)
                log_shifted_profits.append(profit_log_shifted)

            avg_simplified_time = np.mean(simplified_times)
            ci_simplified_time = np.percentile(simplified_times, [2.5, 97.5])

            avg_log_shifted_time = np.mean(log_shifted_times)
            ci_log_shifted_time = np.percentile(log_shifted_times, [2.5, 97.5])

            results.append({
                'Node Size': n,
                'Decimal Size': decim,
                'Simplified Dijkstra Profits': simplified_profits,
                'Log-Shifted Dijkstra Profits': log_shifted_profits,
                'Avg Simplified Time (s)': avg_simplified_time,
                'CI Lower Simplified Time (s)': ci_simplified_time[0],
                'CI Upper Simplified Time (s)': ci_simplified_time[1],
                'Avg Log-Shifted Time (s)': avg_log_shifted_time,
                'CI Lower Log-Shifted Time (s)': ci_log_shifted_time[0],
                'CI Upper Log-Shifted Time (s)': ci_log_shifted_time[1]
            })

    df = pd.DataFrame(results)

    return df


if __name__ == "__main__":
    node_sizes = [5, 10, 15, 20, 30, 40, 50]
    decimal_sizes = [-10, -30, -50, -100, -200]
    n_trials = 100

    df = run_overflow_trials(node_sizes, decimal_sizes, n_trials)
    print(df)
