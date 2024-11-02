from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from typing import List, Tuple, Callable, Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


def brute_force(G: CurrencyGraph,
                start_currency: CurrencyNode,
                node_callback: Callable[[CurrencyNode], None] = None,
                edge_callback: Callable[[CurrencyEdge], None] = None
                ) -> Tuple[List[CurrencyNode], float]:
    """
    Brute force algorithm to find all possible cycsles that start and end at
    the given start_currency in a currency graph while calculating their
    profits.

    ## Attributes
        `G`: The currency graph to be analyzed.
        `start_currency`: The currency node to start the cycle from.
        `node_callback`: A callback function to be called for each node
                         visited.
        `edge_callback`: A callback function to be called for each edge
                         visited.

    ## Returns
        A list of tuples where each tuple contains a list of currency nodes
        that form a cycle and the profit of that cycle.

    ## Example
    ```py
    from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
    nodes = [CurrencyNode('E'), CurrencyNode('D'),
             CurrencyNode('L'), CurrencyNode('F')]
    edges = [CurrencyEdge(nodes[0],nodes[0], 1.0),
             CurrencyEdge(nodes[1],nodes[1], 1.0),
             CurrencyEdge(nodes[2],nodes[2], 1.0),
             CurrencyEdge(nodes[3],nodes[3], 1.0),
             CurrencyEdge(nodes[0],nodes[1], 1.19),
             CurrencyEdge(nodes[1],nodes[0], 0.84),
             CurrencyEdge(nodes[0],nodes[2], 1.33),
             CurrencyEdge(nodes[2],nodes[0], 0.75),
             CurrencyEdge(nodes[0],nodes[3], 1.62),
             CurrencyEdge(nodes[3],nodes[0], 0.62),
             CurrencyEdge(nodes[1],nodes[2], 1.12),
             CurrencyEdge(nodes[2],nodes[1], 0.89),
             CurrencyEdge(nodes[1],nodes[3], 1.37),
             CurrencyEdge(nodes[3],nodes[1], 0.73),
             CurrencyEdge(nodes[2],nodes[3], 1.22),
             CurrencyEdge(nodes[3],nodes[2], 0.82)]
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
                    start_currency: CurrencyNode,
                    node_callback: Callable[[CurrencyNode], None] = None,
                    edge_callback: Callable[[CurrencyEdge], None] = None
                    ) -> Tuple[List[CurrencyNode], float]:
    """
    Brute force algorithm to find all possible cycles that start and end at
    the given start_currency in a currency graph while calculating their
    profits using logarithmic transformation of edge weights to turn
    multiplications into additions.

    ## Parameters
        `G`: The currency graph to be analyzed.
        `start_currency`: The currency node to start the cycle from.
        `node_callback`: A callback function to be called for each node
                         visited.
        `edge_callback`: A callback function to be called for each edge
                         visited.

    ## Returns
        A list of currency nodes representing the most profitable cycle,
        and the associated profit (in its original form, not log-transformed).

    ## Example
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

        if node_callback:
            node_callback(current_node)

        for edge in G.get_edges_from_source(current_node):
            next_node = edge.target
            new_log_profit = current_log_profit + np.log(edge.weight)

            if edge_callback:
                edge_callback(edge)

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
    output = simplified_dijkstra(G, nodes[0], 3)
    ```
    """

    lambda_values = {node: 0.0 for node in G.nodes}
    lambda_values[start_currency] = 1.0

    best_paths = {node: [start_currency] for node in G.nodes}

    if node_callback:
        node_callback(start_currency)

    for edge in G.get_edges_from_source(start_currency):
        lambda_values[edge.target] = edge.weight
        best_paths[edge.target] = [start_currency, edge.target]

        if edge_callback:
            edge_callback(edge)

    for k in range(1, n_passages):
        if verbose:
            print(f"Passage {k}")

        temp_lambda_values = lambda_values.copy()
        temp_best_paths = best_paths.copy()

        for node in G.nodes:
            if node_callback:
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

                    if edge_callback:
                        edge_callback(CurrencyEdge(target_node, node,
                                                   edge_weight))

            temp_lambda_values[node] = max_value
            temp_best_paths[node] = best_path

        lambda_values = temp_lambda_values
        best_paths = temp_best_paths

    final_cycle = (best_paths[start_currency]
                   if best_paths[start_currency][-1] == start_currency
                   else best_paths[start_currency] + [start_currency])

    max_profit = lambda_values[start_currency]

    return final_cycle, max_profit


def log_shifted_simplified_dijkstra(G: CurrencyGraph,
                                    start_currency: CurrencyNode,
                                    n_passages: int,
                                    node_callback: Callable[[CurrencyNode],
                                                            None] = None,
                                    edge_callback: Callable[[CurrencyEdge],
                                                            None] = None,
                                    return_log_profit: bool = False,
                                    verbose: bool = False
                                    ) -> Tuple[List[CurrencyNode], float]:
    """
    A modified Dijkstra algorithm that uses logarithms to avoid underflow
    when handling very small edge weights in a currency graph.

    ## Parameters
        `G`: The currency graph to analyze.
        `start_currency`: The currency node to start and end the cycle from.
        `n_passages`: The maximum number of exchanges (passages) allowed.
        `node_callback`: A callback function to be called for each node
                         visited.
        `edge_callback`: A callback function to be called for each edge
                         visited.
        `return_log_profit`: If True, return the profit in log-transformed
                             form.
        `verbose`: If True, print the progress of the algorithm.

    ## Returns
        The most profitable cycle (list of CurrencyNodes) and
        the corresponding profit.

    ## Example
    ```py
    from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
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
    output = simplified_dijkstra(G, nodes[0], 3)
    ```
    """

    min_edge_weight = min(edge.weight for edge in G.edges)
    epsilon = abs(np.log(min_edge_weight))

    lambda_values = {node: -float('inf') for node in G.nodes}
    lambda_values[start_currency] = 0.0

    best_paths = {node: [start_currency] for node in G.nodes}

    if node_callback:
        node_callback(start_currency)

    for edge in G.get_edges_from_source(start_currency):
        log_weight = np.log(edge.weight) + epsilon
        lambda_values[edge.target] = log_weight
        best_paths[edge.target] = [start_currency, edge.target]

        if edge_callback:
            edge_callback(edge)

    for k in range(1, n_passages):
        if verbose:
            print(f"Passage {k}")

        temp_lambda_values = lambda_values.copy()
        temp_best_paths = best_paths.copy()

        for node in G.nodes:
            if node_callback:
                node_callback(node)

            if verbose:
                print(f"\tNode {node}")
            max_value = lambda_values[node]
            best_path = best_paths[node]

            for target_node in G.nodes:
                edge_weight = G.get_edge_weight(target_node, node)
                if edge_weight > 0:
                    log_weight = np.log(edge_weight) + epsilon
                    new_value = lambda_values[target_node] + log_weight

                    if verbose:
                        print(f"\t\t{target_node} -> {node}: "
                              f"{lambda_values[target_node]} + "
                              f"log({edge_weight}) + {epsilon} "
                              f"= {new_value}")
                    if new_value > max_value:
                        max_value = new_value
                        best_path = best_paths[target_node] + [node]

                    if edge_callback:
                        edge_callback(CurrencyEdge(target_node, node,
                                                   edge_weight))

            temp_lambda_values[node] = max_value
            temp_best_paths[node] = best_path

        lambda_values = temp_lambda_values
        best_paths = temp_best_paths

    final_cycle = (best_paths[start_currency]
                   if best_paths[start_currency][-1] == start_currency
                   else best_paths[start_currency] + [start_currency])

    max_profit = (
        np.exp(lambda_values[start_currency] - (len(final_cycle) - 1)
               * epsilon) if not return_log_profit else
        lambda_values[start_currency])

    return final_cycle, max_profit


def visualize_path(G: CurrencyGraph,
                   algorithm: Callable[[CurrencyGraph,
                                        CurrencyNode,
                                        Optional[int]],
                                       Tuple[List[CurrencyNode], float]],
                   start_currency: CurrencyNode,
                   n_passages: Optional[int] = None,
                   delay: float = 0.5
                   ) -> None:
    """
    Visualize the path found by the given algorithm in the given currency
    graph.

    ## Parameters
        `G`: The currency graph to analyze.
        `algorithm`: The algorithm to use to find the path.
        `start_currency`: The currency node to start the cycle from.
        `n_passages`: The maximum number of exchanges (passages) allowed.
        `delay`: The delay between each step of the visualization.

    ## Example
    ```py
    from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
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
    visualize_path(G, simplified_dijkstra, nodes[0], 3)
    ```
    """

    G_nx = nx.DiGraph()
    for node in G.nodes:
        G_nx.add_node(node.name)
    for edge in G.edges:
        G_nx.add_edge(edge.source.name, edge.target.name, weight=edge.weight)

    for node in G_nx.nodes:
        G_nx.nodes[node]['color'] = 'gray'
        G_nx.nodes[node]['visit_count'] = 0
    for edge in G_nx.edges:
        G_nx[edge[0]][edge[1]]['color'] = 'gray'
        G_nx[edge[0]][edge[1]]['visit_count'] = 0

    visit_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan',
                    'magenta', 'yellow', 'pink']

    def update_display():
        pos = {
            G.nodes[0].name: (0, 0),
            G.nodes[1].name: (1, 0),
            G.nodes[2].name: (1, 1),
            G.nodes[3].name: (0, 1),
        }
        node_colors = [G_nx.nodes[node].get('color', 'gray')
                       for node in G_nx.nodes]
        edge_colors = [G_nx[u][v].get('color', 'gray')
                       for u, v in G_nx.edges]

        plt.clf()
        nx.draw(G_nx, pos, with_labels=True, node_color=node_colors,
                edge_color=edge_colors, node_size=3000, font_size=10,
                font_weight='bold', connectionstyle="arc3,rad=0.2",
                arrows=True)
        edge_labels = {(edge.source.name, edge.target.name): f'{edge.weight}'
                       for edge in G.edges}
        nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels,
                                     label_pos=0.3, font_color='black')
        plt.title(f"Algorithm: {algorithm.__name__}")
        plt.pause(0.01)

    def node_callback(node: CurrencyNode) -> None:
        G_nx.nodes[node.name]['visit_count'] += 1
        visit_count = G_nx.nodes[node.name]['visit_count']
        color_index = min(visit_count - 1, len(visit_colors) - 1)
        G_nx.nodes[node.name]['color'] = visit_colors[color_index]
        update_display()
        time.sleep(delay)

    def edge_callback(edge: CurrencyEdge) -> None:
        G_nx[edge.source.name][edge.target.name]['visit_count'] += 1
        visit_count = G_nx[edge.source.name][edge.target.name]['visit_count']
        color_index = min(visit_count - 1, len(visit_colors) - 1)
        G_nx[edge.source.name][edge.target.name]['color'] = (
            visit_colors[color_index])
        update_display()
        time.sleep(delay)

    if n_passages is not None:
        path, profit = algorithm(G, start_currency, n_passages,
                                 node_callback=node_callback,
                                 edge_callback=edge_callback)
    else:
        path, profit = algorithm(G, start_currency,
                                 node_callback=node_callback,
                                 edge_callback=edge_callback)

    update_display()
    plt.show(block=False)

    print(f"Path found: {[node.name for node in path]}")
    print(f"Associated rate: {profit}")


if __name__ == "__main__":

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

    # visualize_path(G, simplified_dijkstra, nodes[0], 3)
    visualize_path(G, brute_force, nodes[0])
