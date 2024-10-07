from src.currencygraph import CurrencyGraph, CurrencyNode
from typing import List, Tuple


def brute_force(G: CurrencyGraph,
                start_currency: CurrencyNode
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
    nodes = [CurrencyNode('E'), CurrencyNode('D'), CurrencyNode('L')]
    edges = [CurrencyEdge(nodes[0], nodes[1], 2),
             CurrencyEdge(nodes[1], nodes[0], 10),
             CurrencyEdge(nodes[1], nodes[2], 3.5),
             CurrencyEdge(nodes[2], nodes[1], 1/4),
             CurrencyEdge(nodes[2], nodes[0], 12),
             CurrencyEdge(nodes[0], nodes[2], 13)]
    G = CurrencyGraph(nodes, edges)
    print(brute_force(G, nodes[0]))
    ```
    """

    stack = [(start_currency, [start_currency], 1.0)]
    all_cycles_with_profits = []

    while stack:
        current_node, path, current_profit = stack.pop()

        for edge in G.get_edges_from_source(current_node):
            next_node = edge.target
            new_profit = current_profit * edge.weight

            if next_node == start_currency and len(path) > 1:
                all_cycles_with_profits.append((path + [start_currency],
                                                new_profit))
            elif next_node not in path:
                stack.append((next_node, path + [next_node], new_profit))

    return max(all_cycles_with_profits, key=lambda x: x[1])
