from src.currencygraph import CurrencyGraph, CurrencyNode
from typing import List


def brute_force(G: CurrencyGraph,
                target_currency: CurrencyNode) -> List[CurrencyNode]:

    """
    Brute force algorithm to find the more profitable path in a currency graph.

    ## Attributes:
        `G`: CurrencyGraph
            The currency graph to be analyzed.
        `target_currency`: The target currency node to be reached
                           (starting node too).*

    ## Returns:
        A list of CurrencyNode objects representing the optimal path.
    """
    pass
