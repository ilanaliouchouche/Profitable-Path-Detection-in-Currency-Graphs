from fastapi import FastAPI
from pydantic import BaseModel
from src.optimal_paths import brute_force, log_brute_force, simplified_dijkstra
from src.currencygraph import CurrencyGraph, CurrencyNode, CurrencyEdge
from typing import List, Optional, Callable

app = FastAPI()


class CurrencyEdgeModel(BaseModel):
    source: str
    target: str
    weight: float


class CurrencyNodeModel(BaseModel):
    name: str


class CurrencyGraphModel(BaseModel):
    nodes: List[CurrencyNodeModel]
    edges: List[CurrencyEdgeModel]


@app.post("/brute_force/")
def brute_force_endpoint(graph: CurrencyGraphModel, start_currency: str):
    nodes = [CurrencyNode(node.name) for node in graph.nodes]
    edges = [CurrencyEdge(CurrencyNode(edge.source), CurrencyNode(edge.target),
                          edge.weight)
             for edge in graph.edges]
    G = CurrencyGraph(nodes, edges)

    start_node = CurrencyNode(start_currency)

    cycle, profit = brute_force(G, start_node)

    cycle_names = [node.name for node in cycle]
    return {"cycle": cycle_names, "profit": profit}


@app.post("/log_brute_force/")
def log_brute_force_endpoint(graph: CurrencyGraphModel, start_currency: str):
    nodes = [CurrencyNode(node.name) for node in graph.nodes]
    edges = [CurrencyEdge(CurrencyNode(edge.source), CurrencyNode(edge.target),
                          edge.weight)
             for edge in graph.edges]
    G = CurrencyGraph(nodes, edges)

    start_node = CurrencyNode(start_currency)

    cycle, profit = log_brute_force(G, start_node)

    cycle_names = [node.name for node in cycle]
    return {"cycle": cycle_names, "profit": profit}


@app.post("/simplified_dijkstra/")
def simplified_dijkstra_endpoint(graph: CurrencyGraphModel,
                                 start_currency: str, n_passages: int):
    nodes = [CurrencyNode(node.name) for node in graph.nodes]
    edges = [CurrencyEdge(CurrencyNode(edge.source),
                          CurrencyNode(edge.target),
                          edge.weight)
             for edge in graph.edges]
    G = CurrencyGraph(nodes, edges)

    start_node = CurrencyNode(start_currency)

    cycle, profit = simplified_dijkstra(G, start_node, n_passages)

    cycle_names = [node.name for node in cycle]
    return {"cycle": cycle_names, "profit": profit}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,
                host='localhost',
                port=8000)
