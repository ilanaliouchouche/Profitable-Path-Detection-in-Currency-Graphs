import unittest
from src.currencygraph import CurrencyNode, CurrencyEdge, CurrencyGraph


class TestCurrencyNode(unittest.TestCase):

    def test_currency_node_equality(self):
        """
        Test that two CurrencyNode objects are equal
        if they have the same name.
        """

        node1 = CurrencyNode("USD")
        node2 = CurrencyNode("USD")
        node3 = CurrencyNode("EUR")

        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)


class TestCurrencyEdge(unittest.TestCase):

    def test_currency_edge_equality(self):
        """
        Test that two CurrencyEdge objects are equal
        if they have the same source and target
        """

        node1 = CurrencyNode("USD")
        node2 = CurrencyNode("EUR")
        edge1 = CurrencyEdge(node1, node2, 0.8)
        edge2 = CurrencyEdge(node1, node2, 0.8)
        edge3 = CurrencyEdge(node1, node2, 0.9)

        self.assertEqual(edge1, edge2)
        self.assertEqual(edge1, edge3)


class TestCurrencyGraph(unittest.TestCase):

    def setUp(self):
        self.node_usd = CurrencyNode("USD")
        self.node_eur = CurrencyNode("EUR")
        self.node_gbp = CurrencyNode("GBP")
        self.edge1 = CurrencyEdge(self.node_usd, self.node_eur, 0.8)
        self.edge2 = CurrencyEdge(self.node_eur, self.node_gbp, 1.1)
        self.graph = CurrencyGraph([self.node_usd, self.node_eur],
                                   [self.edge1])

    def test_graph_initialization(self):
        """
        Test that a CurrencyGraph object is initialized
        with the correct nodes and edges.
        """

        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(len(self.graph.edges), 1)

    def test_get_adjacency_matrix(self):
        """
        Test that the adjacency matrix of the graph is
        correctly computed.
        """

        adj_matrix = self.graph.get_adjacency_matrix()
        self.assertEqual(adj_matrix.shape, (2, 2))
        self.assertEqual(adj_matrix[0, 1], 0.8)
        self.assertEqual(adj_matrix[1, 0], 0)

        adj_matrix_df = self.graph.get_adjacency_matrix(as_dataframe=True)
        self.assertEqual(adj_matrix_df.loc["USD", "EUR"], 0.8)
        self.assertEqual(adj_matrix_df.loc["EUR", "USD"], 0)

    def test_add_node(self):
        """
        Test that a node can be added to the graph.
        """

        self.graph.add_node(self.node_gbp)
        self.assertIn(self.node_gbp, self.graph.nodes)
        self.assertEqual(len(self.graph.nodes), 3)

        with self.assertRaises(ValueError):
            self.graph.add_node(self.node_gbp)

    def test_add_edge(self):
        """
        Test that an edge can be added to the graph.
        """

        self.graph.add_node(self.node_gbp)
        self.graph.add_edge(self.edge2)
        self.assertIn(self.edge2, self.graph.edges)
        self.assertEqual(len(self.graph.edges), 2)

        with self.assertRaises(ValueError):
            self.graph.add_edge(self.edge2)

    def test_remove_node(self):
        """
        Test that a node can be removed from the graph.
        """

        self.graph.remove_node(self.node_usd)
        self.assertNotIn(self.node_usd, self.graph.nodes)
        self.assertEqual(len(self.graph.nodes), 1)
        self.assertEqual(len(self.graph.edges), 0)

        with self.assertRaises(ValueError):
            self.graph.remove_node(self.node_gbp)

    def test_remove_edge(self):
        """
        Test that an edge can be removed from the graph.
        """

        self.graph.remove_edge(self.edge1)
        self.assertNotIn(self.edge1, self.graph.edges)
        self.assertEqual(len(self.graph.edges), 0)

        with self.assertRaises(ValueError):
            self.graph.remove_edge(self.edge2)

    def test_graph_setters(self):
        """
        Test that the nodes and edges of the graph can be set.
        """

        new_nodes = [self.node_usd, self.node_gbp]
        self.graph.nodes = new_nodes
        self.assertEqual(self.graph.nodes, new_nodes)

        new_edges = [CurrencyEdge(self.node_usd, self.node_gbp, 0.5)]
        self.graph.edges = new_edges
        self.assertEqual(self.graph.edges, new_edges)


if __name__ == "__main__":
    unittest.main()
