import unittest
from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from src.optimal_paths import (brute_force,
                               log_brute_force,
                               simplified_dijkstra,
                               log_shifted_simplified_dijkstra)
import numpy as np


class TestOptimalPaths(unittest.TestCase):

    def setUp(self):
        self.nodes = [CurrencyNode('E'), CurrencyNode('D'),
                      CurrencyNode('L'), CurrencyNode('F')]

        self.edges = [CurrencyEdge(self.nodes[0], self.nodes[0], 1.0),
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

        self.G = CurrencyGraph(self.nodes, self.edges)

        self.output_path = [CurrencyNode(name='E'),
                            CurrencyNode(name='D'),
                            CurrencyNode(name='F'),
                            CurrencyNode(name='E')]

        self.product_profit = 1.010786

    def test_brute_force(self):
        """
        Test the brute force algorithm to find the most profitable cycle in a
        currency graph.
        """

        output = brute_force(self.G, self.nodes[0])

        self.assertEqual(output[0], self.output_path)
        self.assertTrue(np.isclose(output[1], self.product_profit, rtol=1e-6))

    def test_log_brute_force(self):
        """
        Test the log brute force algorithm to find the most profitable cycle in
        a currency graph.
        """

        output = log_brute_force(self.G, self.nodes[0])

        self.assertEqual(output[0], self.output_path)
        self.assertTrue(np.isclose(output[1], self.product_profit, rtol=1e-6))

    def test_simplified_dijkstra(self):
        """
        Test the simplified Dijkstra algorithm to find the most profitable
        cycle in a currency graph.
        """

        output = simplified_dijkstra(self.G, self.nodes[0], 3)

        self.assertEqual(output[0], self.output_path)
        self.assertTrue(np.isclose(output[1], self.product_profit, rtol=1e-6))

    def test_log_shifted_simplified_dijkstra(self):
        """
        Test the log shifted simplified Dijkstra algorithm to find the most
        profitable cycle in a currency graph.
        """

        output = log_shifted_simplified_dijkstra(self.G, self.nodes[0], 3)

        self.assertEqual(output[0], self.output_path)
        self.assertTrue(np.isclose(output[1], self.product_profit, rtol=1e-6))
