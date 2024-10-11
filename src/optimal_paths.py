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
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(self.nodes[0], self.nodes[1], 1.19),
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

        for edge in G.get_edges_from_source(current_node):
            next_node = edge.target
            new_profit = current_profit * edge.weight

            if next_node == start_currency and len(path) > 1:
                all_cycles_with_profits.append((path + [start_currency],
                                                new_profit))
            elif next_node not in path:
                stack.append((next_node, path + [next_node], new_profit))

    return max(all_cycles_with_profits, key=lambda x: x[1])


def simplified_dijkstra(G: CurrencyGraph,
                        start_currency: CurrencyNode,
                        n_passages: int
                        ) -> Tuple[List[CurrencyNode], float]:
    """
    A crafted dijkstra algorithm to find the most profitable cycle that
    starts and ends at the given start_currency in a currency graph, with a
    limit on the number of exchanges (n_passages).

    ## Parameters
        `G`: The currency graph to analyze.
        `start_currency`: The currency node to start and end the cycle from.
        `n_passages`: The maximum number of exchanges (passages) allowed.

    ## Returns
        The most profitable cycle (list of CurrencyNodes)
        and the corresponding profit.

    ## Example
    ```py
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(self.nodes[0], self.nodes[1], 1.19),
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

    for edge in G.get_edges_from_source(start_currency):
        lambda_values[edge.target] = edge.weight
        best_paths[edge.target] = [start_currency, edge.target]

    for k in range(1, n_passages):
        print(f"Passage {k}")
        for node in G.nodes:
            print(f"\tNode {node}")
            max_value = lambda_values[node]
            best_path = best_paths[node]
            for target_node in G.nodes:
                new_value = (lambda_values[target_node] *
                             G.get_edge_weight(target_node, node))
                print(f"\t\t{target_node} -> {node}: {new_value}")
                if new_value > max_value:
                    max_value = new_value
                    best_path = best_paths[node] + [node]
            print(f"\tMax value: {max_value} of path {best_path}")
            lambda_values[node] = max_value
            best_paths[node] = best_path

    final_cycle = best_paths[start_currency] + [start_currency]
    max_profit = lambda_values[start_currency]

    return final_cycle, max_profit


def simplified_dijkstrv(G: CurrencyGraph,
                        start_currency: CurrencyNode,
                        n_passages: int
                        ) -> Tuple[List[CurrencyNode], float]:
    """
    A crafted dijkstra algorithm to find the most profitable cycle that
    starts and ends at the given start_currency in a currency graph, with a
    limit on the number of exchanges (n_passages).

    ## Parameters
        G: The currency graph to analyze.
        start_currency: The currency node to start and end the cycle from.
        n_passages: The maximum number of exchanges (passages) allowed.

    ## Returns
        The most profitable cycle (list of CurrencyNodes)
        and the corresponding profit.
    """

    # Step 1 : Define a map dictionary to map each node to an index
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    # Step 2 :Initialize lambda iterations
    lambda_iterations = [[0.0] * n_passages for _ in range(len(G.nodes))]
    lambda_values = {node: 0.0 for node in G.nodes}

    # Step 3 : Initialize the lambda values for the start node
    lambda_values[start_currency] = 1.0
    start_index = node_to_index[start_currency]
    lambda_iterations[start_index][0] = 1.0

    # Step 4 : Initialize the lambda values for the rest of the nodes
    for edge in G.get_edges_from_source(start_currency):
        lambda_values[edge.target] = edge.weight
        target_index = node_to_index[edge.target]
        lambda_iterations[target_index][0] = edge.weight

    # Step 5 : Iterate over the number of passages
    for k in range(1, n_passages):
        # Step 6 : Iterate over the nodes
        print(f"Passage {k}")
        for node in G.nodes:
            print(f"\tNode {node}")
            # Get the index of the node
            j = node_to_index[node]
            # update the lambda j,k value
            lambda_nk = lambda_iterations[j][k]
            for target_node in G.nodes:
                node_index = node_to_index[target_node]
                last_lambda = lambda_iterations[node_index][k - 1]
                tij = G.get_edge_weight(node, target_node)
                lambda_nk = max(lambda_nk, last_lambda * tij)
                print(f"\t\t{node} -> {edge.target}: {lambda_nk}")
            lambda_iterations[j][k] = lambda_nk
            lambda_values[node] = lambda_nk

    # Step 7 : Find the most profitable cycle from the source node
    max_profit_cycle = []
    max_profit = lambda_values[start_currency]
    for node in G.nodes:
        if node != start_currency:
            node_index = node_to_index[node]
            last_lambda = lambda_iterations[node_index][n_passages - 1]
            if last_lambda * G.get_edge_weight(node, start_currency) > max_profit:
                max_profit = last_lambda * G.get_edge_weight(node, start_currency)
                max_profit_cycle = [node, start_currency]

    return max_profit_cycle, max_profit
