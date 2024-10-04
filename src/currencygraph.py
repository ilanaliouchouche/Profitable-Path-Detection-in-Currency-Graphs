import numpy as np
import pandas as pd
from typing import Sequence, Union, List
from dataclasses import dataclass


@dataclass
class CurrencyNode:
    """
    A node in the currency graph.

    ## Attributes
        `name`: The name of the currency
        `date`: (Optional) The date of the currency rate.

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

        return self.name == other.name


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

        return list(self._nodes)

    @property
    def edges(self) -> List[CurrencyEdge]:

        return list(self._edges)

    @nodes.setter
    def nodes(self, nodes: List[CurrencyNode]) -> None:

        self._nodes = nodes

    @edges.setter
    def edges(self, edges: List[CurrencyEdge]) -> None:

        self._edges = edges

    def __repr__(self) -> str:

        return f"CurrencyGraph(nodes={self._nodes}, edges={self._edges})"

    def __str__(self) -> str:

        return self.__repr__()
