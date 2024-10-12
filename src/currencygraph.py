import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Sequence, Union, List
from dataclasses import dataclass


@dataclass
class CurrencyNode:
    """
    A node in the currency graph.

    ## Attributes
        `name`: The name of the currency

    ## Example
    ```py
    from currencygraph import CurrencyNode

    node_usd = CurrencyNode('USD')
    node_eu = CurrencyNode('EUR')
    ```
    """

    name: str

    def __eq__(self,
               other: 'CurrencyNode') -> bool:
        """
        Checks if two CurrencyNode objects are equal.

        ## Parameters
            `other`: The other CurrencyNode object to compare.

        ## Returns
            A boolean indicating whether the two CurrencyNodes are equal.

        ## Example
        ```py
        from currencygraph import CurrencyNode

        node_usd = CurrencyNode('USD')
        node_eu = CurrencyNode('EUR')
        print(node_usd == node_eu)  # False
        ```
        """

        return self.name == other.name

    def __hash__(self) -> int:
        """
        Returns the hash value of the CurrencyNode object.

        ## Returns
            The hash value of the CurrencyNode object.

        ## Example
        ```py
        from currencygraph import CurrencyNode

        node_usd = CurrencyNode('USD')
        print(hash(node_usd))
        ```
        """

        return hash(self.name)


@dataclass
class CurrencyEdge:
    """
    An edge in the currency graph.

    ## Attributes
        `source`: The source currency node.
        `target`: The target currency node.
        `weight`: The weight of the edge.

    ## Example
    ```py
    from currencygraph import CurrencyNode, CurrencyEdge

    source = CurrencyNode('USD')
    target = CurrencyNode('EUR')
    edge = CurrencyEdge(source, target, 0.8)
    ```
    """

    source: CurrencyNode
    target: CurrencyNode
    weight: float

    def __eq__(self,
               other: 'CurrencyEdge') -> bool:
        """
        Checks if two CurrencyEdge objects are equal.

        ## Parameters
            `other`: The other CurrencyEdge object to compare.

        ## Returns
            A boolean indicating whether the two CurrencyEdges are equal.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge

        source = CurrencyNode('USD')
        target = CurrencyNode('EUR')
        edge = CurrencyEdge(source, target, 0.8)
        edge2 = CurrencyEdge(source, target, 0.8)
        print(edge == edge2)  # True
        ```
        """

        return self.source == other.source and self.target == other.target


class CurrencyGraph:
    """
    A graph representing currency rates between different currencies.
    """

    def __init__(self,
                 nodes: Sequence[CurrencyNode],
                 edges: Sequence[CurrencyEdge]) -> None:
        """
        Constructor for the CurrencyGraph class.

        ## Parameters
            `nodes`: A sequence of CurrencyNode objects representing the nodes
                     of the graph.
            `edges`: A sequence of CurrencyEdge objects representing the edges
                     of the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        nodes = [CurrencyNode('USD'), CurrencyNode('EUR')]
        edges = [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)]
        graph = CurrencyGraph(nodes, edges)
        ```
        """

        self._nodes = nodes if isinstance(nodes, list) else list(nodes)
        self._edges = edges if isinstance(edges, list) else list(edges)

        # We check if the nodes in the _edges are present in the _nodes
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise ValueError(f"Node {edge.source} not found in the nodes.")
            if edge.target not in self._nodes:
                raise ValueError(f"Node {edge.target} not found in the nodes.")

        # We check if the nodes in the _nodes are unique
        if len(set([node.name for node in self._nodes])) != len(self._nodes):
            raise ValueError("Nodes must be unique.")

        # We check if the edges in the _edges are unique
        if len(set([(edge.source.name, edge.target.name)
                    for edge in self._edges])) != len(self._edges):
            raise ValueError("Edges must be unique.")

    def get_adjacency_matrix(self,
                             as_dataframe: bool = False
                             ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Returns the adjacency matrix of the graph.

        ## Parameters
            `as_dataframe`: A boolean flag indicating whether to return
                            the adjacency matrix as a pandas DataFrame.

        ## Returns
            A numpy array representing the adjacency matrix of the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        nodes = [CurrencyNode('USD'), CurrencyNode('EUR')]
        edges = [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)]
        graph = CurrencyGraph(nodes, edges)

        adj_matrix = graph.get_adjacency_matrix(as_dataframe=False)
        print(adj_matrix)
        ```
        """

        n = len(self._nodes)
        adj_matrix = np.zeros((n, n))
        for edge in self._edges:
            source_idx = self._nodes.index(edge.source)
            target_idx = self._nodes.index(edge.target)
            adj_matrix[source_idx, target_idx] = edge.weight

        if as_dataframe:
            return pd.DataFrame(adj_matrix,
                                index=[node.name for node in self._nodes],
                                columns=[node.name for node in self._nodes])

        return adj_matrix

    def get_edge_weight(self,
                        source: CurrencyNode,
                        target: CurrencyNode) -> float:
        """
        Returns the weight of the edge between two nodes.

        ## Parameters
            `source`: The source node of the edge.
            `target`: The target node of the edge.

        ## Returns
            The weight of the edge between the source and target nodes.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        nodes = [CurrencyNode('USD'), CurrencyNode('EUR')]
        edges = [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)]
        graph = CurrencyGraph(nodes, edges)
        weight = graph.get_edge_weight(CurrencyNode('USD'),
        CurrencyNode('EUR'))
        print(weight)
        ```
        """

        for edge in self._edges:
            if edge.source == source and edge.target == target:
                return edge.weight

        raise ValueError(f"Edge from {source} to {target} not found.")

    def get_edges_from_source(self,
                              node: CurrencyNode) -> List[CurrencyEdge]:
        """
        Returns the edges from a source node.

        ## Parameters
            `node`: The source node to get the edges from.

        ## Returns
            A list of CurrencyEdge objects representing the edges from the
            source node.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        nodes = [CurrencyNode('USD'), CurrencyNode('EUR')]
        edges = [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)]
        graph = CurrencyGraph(nodes, edges)

        edges = graph.get_edges_from_source(CurrencyNode('USD'))
        print(edges)
        ```
        """

        return [edge for edge in self._edges if edge.source == node]

    def show(self) -> None:
        """
        Displays the currency graph using NetworkX and Matplotlib.

        ## Example
        ```py
        graph = CurrencyGraph(nodes, edges)
        graph.show()
        ```
        """
        G = nx.DiGraph()

        for node in self._nodes:
            G.add_node(node.name)
        for edge in self._edges:
            G.add_edge(edge.source.name, edge.target.name, weight=edge.weight)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000,
                node_color='lightblue', font_size=10,
                font_weight='bold', arrows=True)
        edge_labels = {(edge.source.name, edge.target.name): f'{edge.weight}'
                       for edge in self._edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()

    def add_node(self,
                 node: CurrencyNode) -> None:
        """
        Adds a node to the graph.

        ## Parameters
            `node`: The CurrencyNode object to add to the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyGraph

        graph = CurrencyGraph([], [])
        graph.add_node(CurrencyNode('USD'))
        ```
        """

        if node not in self._nodes:
            self._nodes.append(node)
        else:
            raise ValueError(f"Node {node} already exists in the graph.")

    def add_edge(self,
                 edge: CurrencyEdge) -> None:
        """
        Adds an edge to the graph.

        ## Parameters
            `edge`: The CurrencyEdge object to add to the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'),CurrencyNode('EUR')], [])
        graph.add_edge(CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'),
        0.8))
        ```
        """

        if edge.source not in self._nodes:
            raise ValueError(f"Node {edge.source} not found in the graph.")
        if edge.target not in self._nodes:
            raise ValueError(f"Node {edge.target} not found in the graph.")
        if edge not in self._edges:
            self._edges.append(edge)
        else:
            raise ValueError(f"Edge {edge} already exists in the graph.")

    def remove_edge(self,
                    edge: CurrencyEdge) -> None:
        """
        Removes an edge from the graph.

        ## Parameters
            `edge`: The CurrencyEdge object to remove from the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'),CurrencyNode('EUR')],
        [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)])
        graph.remove_edge(CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR')
        , 0.8))
        ```
        """

        if edge in self._edges:
            self._edges.remove(edge)
        else:
            raise ValueError(f"Edge {edge} not found in the graph.")

    def remove_node(self,
                    node: CurrencyNode) -> None:
        """
        Removes a node from the graph.

        ## Parameters
            `node`: The CurrencyNode object to remove from the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'),CurrencyNode('EUR')],
        [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)])
        graph.remove_node(CurrencyNode('USD'))
        ```
        """

        if node in self._nodes:
            self._nodes.remove(node)
            self._edges = [edge for edge in self._edges
                           if edge.source != node and edge.target != node]
        else:
            raise ValueError(f"Node {node} not found in the graph.")

    @property
    def nodes(self) -> List[CurrencyNode]:
        """
        Returns the nodes of the graph.

        ## Returns
            A list of CurrencyNode objects representing the nodes of the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'), CurrencyNode('EUR')], [])
        nodes = graph.nodes
        print(nodes)
        ```
        """

        return self._nodes

    @property
    def edges(self) -> List[CurrencyEdge]:
        """
        Returns the edges of the graph.

        ## Returns
            A list of CurrencyEdge objects representing the edges of the graph.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'), CurrencyNode('EUR')],
        [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)])
        edges = graph.edges
        print(edges)
        ```
        """

        return self._edges

    @nodes.setter
    def nodes(self, nodes: List[CurrencyNode]) -> None:
        """
        Sets the nodes of the graph.

        ## Parameters
            `nodes`: A list of CurrencyNode objects representing the nodes.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyGraph

        graph = CurrencyGraph([], [])
        graph.nodes = [CurrencyNode('USD'), CurrencyNode('EUR')]
        ```
        """

        self._nodes = nodes

    @edges.setter
    def edges(self, edges: List[CurrencyEdge]) -> None:
        """
        Sets the edges of the graph.

        ## Parameters
            `edges`: A list of CurrencyEdge objects representing the edges.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([], [])
        graph.edges = [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR')
        ,0.8)]
        ```
        """

        self._edges = edges

    def __repr__(self) -> str:
        """
        Returns a string representation of the CurrencyGraph object.

        ## Returns
            A string representation of the CurrencyGraph object.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'), CurrencyNode('EUR')],
        [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)])
        print(graph)
        ```
        """

        return f"CurrencyGraph(nodes={self._nodes}, edges={self._edges})"

    def __str__(self) -> str:
        """
        Returns a string representation of the CurrencyGraph object.

        ## Returns
            A string representation of the CurrencyGraph object.

        ## Example
        ```py
        from currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph

        graph = CurrencyGraph([CurrencyNode('USD'), CurrencyNode('EUR')],
        [CurrencyEdge(CurrencyNode('USD'), CurrencyNode('EUR'), 0.8)])
        print(graph)
        ```
        """

        return self.__repr__()
