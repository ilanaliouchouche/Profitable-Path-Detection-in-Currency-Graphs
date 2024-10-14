from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from typing import List, Tuple, Callable
import numpy as np


def brute_force(G: CurrencyGraph,
                start_currency: CurrencyNode,
                node_callback: Callable[[CurrencyNode], None] = None,
                edge_callback: Callable[[CurrencyEdge], None] = None
                ) -> Tuple[List[CurrencyNode], float]:
    """
    Brute force algorithm to find all possible cycles that start and end at
    the given start_currency in a currency graph while calculating their
    profits.

    ## Attributes
        `G`: The currency graph to be analyzed.
        `start_currency`: The currency node to start the cycle from.

    ## Returns
        A list of tuples where each tuple contains a list of currency nodes
        that form a cycle and the profit of that cycle.

    ## Example
    ```py
    from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(self.nodes[0], self.nodes[0], 1.0),
             CurrencyEdge(self.nodes[1], self.nodes[1], 1.0),
             CurrencyEdge(self.nodes[2], self.nodes[2], 1.0),
             CurrencyEdge(self.nodes[3], self.nodes[3], 1.0),
             CurrencyEdge(self.nodes[0], self.nodes[1], 1.19),
             CurrencyEdge(self.nodes[1], self.nodes[0], 0.84),
             CurrencyEdge(self.nodes[0], self.nodes[2], 1.33),
             CurrencyEdge(self.nodes[2], self.nodes[0], 0.75),
             CurrencyEdge(self.nodes[0], self.nodes[3], 1.62),
             CurrencyEdge(self.nodes[3], self.nodes[0], 0.62),
             CurrencyEdge(self.nodes[1], self.nodes[2], 1.12),
             CurrencyEdge(self.nodes[2], self.nodes[1], 0.89),
             CurrencyEdge(self.nodes[1], self.nodes[3], 1.37),
             CurrencyEdge(self.nodes[3], self.nodes[1], 0.73),
             CurrencyEdge(self.nodes[2], self.nodes[3], 1.22),
             CurrencyEdge(self.nodes[3], self.nodes[2], 0.82)]
    G = CurrencyGraph(nodes, edges)
    output = brute_force(G, nodes[0])
    ```
    """

    stack = [(start_currency, [start_currency], 1.0)]
    all_cycles_with_profits = []

    while stack:
        current_node, path, current_profit = stack.pop()

        if node_callback:
            node_callback(current_node)

        for edge in G.get_edges_from_source(current_node):
            next_node = edge.target
            new_profit = current_profit * edge.weight

            if edge_callback:
                edge_callback(edge)

            if next_node == start_currency and len(path) > 1:
                all_cycles_with_profits.append((path + [start_currency],
                                                new_profit))
            elif next_node not in path:
                stack.append((next_node, path + [next_node], new_profit))

    return max(all_cycles_with_profits, key=lambda x: x[1])


def log_brute_force(G: CurrencyGraph,
                    start_currency: CurrencyNode
                    ) -> Tuple[List[CurrencyNode], float]:
    """
    Brute force algorithm to find all possible cycles that start and end at
    the given start_currency in a currency graph while calculating their
    profits using logarithmic transformation of edge weights to turn
    multiplications into additions.

    ## Parameters:
        `G`: The currency graph to be analyzed.
        `start_currency`: The currency node to start the cycle from.

    ## Returns:
        A list of currency nodes representing the most profitable cycle,
        and the associated profit (in its original form, not log-transformed).

    ## Example:
    ```py
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(nodes[0], nodes[0], 1.0),
             CurrencyEdge(nodes[1], nodes[1], 1.0),
             CurrencyEdge(nodes[2], nodes[2], 1.0),
             CurrencyEdge(nodes[3], nodes[3], 1.0),
             CurrencyEdge(nodes[0], nodes[1], 1.19),
             CurrencyEdge(nodes[1], nodes[0], 0.84),
             CurrencyEdge(nodes[0], nodes[2], 1.33),
             CurrencyEdge(nodes[2], nodes[0], 0.75),
             CurrencyEdge(nodes[0], nodes[3], 1.62),
             CurrencyEdge(nodes[3], nodes[0], 0.62),
             CurrencyEdge(nodes[1], nodes[2], 1.12),
             CurrencyEdge(nodes[2], nodes[1], 0.89),
             CurrencyEdge(nodes[1], nodes[3], 1.37),
             CurrencyEdge(nodes[3], nodes[1], 0.73),
             CurrencyEdge(nodes[2], nodes[3], 1.22),
             CurrencyEdge(nodes[3], nodes[2], 0.82)]
    G = CurrencyGraph(nodes, edges)
    output = log_brute_force(G, nodes[0])
    ```
    """

    stack = [(start_currency, [start_currency], 0.0)]
    all_cycles_with_profits = []

    while stack:
        current_node, path, current_log_profit = stack.pop()

        for edge in G.get_edges_from_source(current_node):
            next_node = edge.target
            new_log_profit = current_log_profit + np.log(edge.weight)

            if next_node == start_currency and len(path) > 1:
                all_cycles_with_profits.append((path + [start_currency],
                                                np.exp(new_log_profit)))
            elif next_node not in path:
                stack.append((next_node, path + [next_node], new_log_profit))

    return max(all_cycles_with_profits, key=lambda x: x[1])


def simplified_dijkstra(G: CurrencyGraph,
                        start_currency: CurrencyNode,
                        n_passages: int,
                        node_callback: Callable[[CurrencyNode], None] = None,
                        edge_callback: Callable[[CurrencyEdge], None] = None,
                        verbose: bool = False
                        ) -> Tuple[List[CurrencyNode], float]:
    """
    A crafted dijkstra algorithm to find the most profitable cycle that
    starts and ends at the given start_currency in a currency graph, with a
    limit on the number of exchanges (n_passages).

    ## Parameters
        `G`: The currency graph to analyze.
        `start_currency`: The currency node to start and end the cycle from.
        `n_passages`: The maximum number of exchanges (passages) allowed.
        `node_callback`: A callback function to be called for each node
                         visited.
        `edge_callback`: A callback function to be called for each edge
                         visited.l
        `verbose`: If True, print the progress of the algorithm.

    ## Returns
        The most profitable cycle (list of CurrencyNodes)
        and the corresponding profit.

    ## Example
    ```py
    from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(self.nodes[0], self.nodes[0], 1.0),
             CurrencyEdge(self.nodes[1], self.nodes[1], 1.0),
             CurrencyEdge(self.nodes[2], self.nodes[2], 1.0),
             CurrencyEdge(self.nodes[3], self.nodes[3], 1.0),
             CurrencyEdge(self.nodes[0], self.nodes[1], 1.19),
             CurrencyEdge(self.nodes[1], self.nodes[0], 0.84),
             CurrencyEdge(self.nodes[0], self.nodes[2], 1.33),
             CurrencyEdge(self.nodes[2], self.nodes[0], 0.75),
             CurrencyEdge(self.nodes[0], self.nodes[3], 1.62),
             CurrencyEdge(self.nodes[3], self.nodes[0], 0.62),
             CurrencyEdge(self.nodes[1], self.nodes[2], 1.12),
             CurrencyEdge(self.nodes[2], self.nodes[1], 0.89),
             CurrencyEdge(self.nodes[1], self.nodes[3], 1.37),
             CurrencyEdge(self.nodes[3], self.nodes[1], 0.73),
             CurrencyEdge(self.nodes[2], self.nodes[3], 1.22),
             CurrencyEdge(self.nodes[3], self.nodes[2], 0.82)]
    G = CurrencyGraph(nodes, edges)
    output = simplified_dijkstra(G, nodes[0], 3)
    ```
    """

    lambda_values = {node: 0.0 for node in G.nodes}
    lambda_values[start_currency] = 1.0

    best_paths = {node: [start_currency] for node in G.nodes}

    node_callback(start_currency)

    for edge in G.get_edges_from_source(start_currency):
        lambda_values[edge.target] = edge.weight
        best_paths[edge.target] = [start_currency, edge.target]

        edge_callback(edge)

    for k in range(1, n_passages):
        if verbose:
            print(f"Passage {k}")

        temp_lambda_values = lambda_values.copy()
        temp_best_paths = best_paths.copy()

        for node in G.nodes:
            node_callback(node)

            if verbose:
                print(f"\tNode {node}")
            max_value = lambda_values[node]
            best_path = best_paths[node]

            for target_node in G.nodes:
                edge_weight = G.get_edge_weight(target_node, node)
                if edge_weight > 0:
                    new_value = lambda_values[target_node] * edge_weight
                    if verbose:
                        print(f"\t\t{target_node} -> {node}: "
                              f"{lambda_values[target_node]}*"
                              f"{edge_weight}={new_value}")
                    if new_value > max_value:
                        max_value = new_value
                        best_path = best_paths[target_node] + [node]

                    edge_callback(CurrencyEdge(target_node, node, edge_weight))

            temp_lambda_values[node] = max_value
            temp_best_paths[node] = best_path

        lambda_values = temp_lambda_values
        best_paths = temp_best_paths

    final_cycle = (best_paths[start_currency]
                   if best_paths[start_currency][-1] == start_currency
                   else best_paths[start_currency] + [start_currency])

    max_profit = lambda_values[start_currency]

    return final_cycle, max_profit
