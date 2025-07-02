from __future__ import annotations

from typing import NamedTuple
from functools import reduce
from operator import mul, add
from enum import IntEnum
from collections import defaultdict

import numpy as np
from scipy.special import logit
import networkx as nx
from simanneal import Annealer
import matplotlib.pyplot as plt

metric_aggregators = {"latency": add, "throughput": min, "integrity": mul}


class Opinion(NamedTuple):
    belief: float
    disbelief: float
    uncertainty: float

    def __mul__(self, other: Opinion) -> Opinion:
        if not isinstance(other, Opinion):
            return NotImplemented
        belief = self.belief * other.belief + (self.belief * other.uncertainty + self.uncertainty * other.belief) / 3
        disbelief = self.disbelief + other.disbelief - self.disbelief * other.disbelief
        uncertainty = 1 - belief - disbelief
        return Opinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty)


def random_opinion():
    uncertainty = np.random.rand() * 0.01
    belief = np.random.choice([1, 0.95, 0.9]) * (1 - uncertainty)
    disbelief = 1 - belief - uncertainty
    return Opinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty)


def make_random_network(n: int, p: float) -> nx.Graph:
    G = nx.fast_gnp_random_graph(n, p)
    for u, v in G.edges():
        G[u][v]["latency"] = np.random.rand() * 10
        G[u][v]["throughput"] = np.random.rand() * 10
        G[u][v]["integrity"] = random_opinion()
    return G


def single_metric_path_cost(graph: nx.Graph, path: list[int], metric: str) -> float:
    aggregator = metric_aggregators[metric]
    weights = [graph[u][v][metric] for u, v in zip(path[:-1], path[1:])]
    return reduce(aggregator, weights)


class MoveType(IntEnum):
    REPLACE = 0
    INSERT = 1
    BYPASS = 2


class PathAnnealer(Annealer):
    def __init__(self, state, graph, factors: dict[str, float]) -> None:
        self.graph = graph
        self.data = defaultdict(list)
        self._cur_values = {}
        self.factors = factors
        super(PathAnnealer, self).__init__(state)

    def move(self):
        path_len = len(self.state)
        if path_len < 3:
            return

        move_type = np.random.choice(list(MoveType))

        if move_type == MoveType.REPLACE:
            idx = np.random.randint(0, path_len - 2)
            s = self.state[idx]
            m = self.state[idx + 1]
            t = self.state[idx + 2]
            pool = set(self.graph.neighbors(s)) & set(self.graph.neighbors(t)) - {m} - set(self.state)
            if not pool:
                return
            new_m = int(np.random.choice(list(pool)))
            self.state[idx + 1] = new_m

        elif move_type == MoveType.INSERT:
            idx = np.random.randint(0, path_len - 1)
            s = self.state[idx]
            t = self.state[idx + 1]
            pool = set(self.graph.neighbors(s)) & set(self.graph.neighbors(t)) - set(self.state)
            if not pool:
                return
            new_m = int(np.random.choice(list(pool)))
            self.state.insert(idx + 1, new_m)

        elif move_type == MoveType.BYPASS:
            idx = np.random.randint(0, path_len - 2)
            s = self.state[idx]
            m = self.state[idx + 1]
            t = self.state[idx + 2]
            if s in set(self.graph.neighbors(t)):
                self.state.pop(idx + 1)

        if len(self.state) > len(set(self.state)):
            raise ValueError("Duplicate nodes in path.")

    def energy(self) -> float:
        graph = self.graph
        path = self.state

        latency_cost = single_metric_path_cost(graph, path, "latency")
        self._cur_values["path_latency"] = latency_cost

        throughput_cost = single_metric_path_cost(graph, path, "throughput")
        self._cur_values["path_throughput"] = throughput_cost

        integrity_cost = Opinion(*single_metric_path_cost(graph, path, "integrity"))
        self._cur_values["path_integrity"] = integrity_cost

        total_cost = (
            self.factors["latency"] * latency_cost
            + self.factors["throughput"] * throughput_cost
            + self.factors["integrity"] * logit(integrity_cost.belief)
        )
        self._cur_values["path_total_cost"] = total_cost

        return total_cost

    def update(self, step, T, E, acceptance, improvement):
        # if self.accept is True:
        self.data["energy"].append(E)
        self.data["step"].append(step)
        self.data["temperature"].append(T)
        self.data["path_length"].append(len(self.state))
        for key, value in self._cur_values.items():
            self.data[key].append(value)
        self.default_update(step, T, E, acceptance, improvement)

    def plot(self, axs: list[plt.Axes] | None = None):
        if axs is None:
            fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        else:
            fig = axs[0, 0].figure

        ax1 = axs[0, 0]
        ax1.plot(self.data["step"], self.data["energy"], color="tab:blue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Energy", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(self.data["step"], self.data["temperature"], color="tab:orange", linestyle="--")
        ax2.set_ylabel("Temperature", color="tab:orange")
        ax2.set_yscale("log")

        axs[0, 1].plot(self.data["step"], self.data["path_latency"])
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Path Latency")

        axs[1, 0].plot(self.data["step"], self.data["path_throughput"])
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Path Throughput")

        integrity_belief = [op.belief for op in self.data["path_integrity"]]
        integrity_disbelief = [op.disbelief for op in self.data["path_integrity"]]
        integrity_uncertainty = [op.uncertainty for op in self.data["path_integrity"]]
        axs[1, 1].plot(self.data["step"], integrity_belief, label="Belief")
        axs[1, 1].plot(self.data["step"], integrity_disbelief, label="Disbelief")
        axs[1, 1].plot(self.data["step"], integrity_uncertainty, label="Uncertainty")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Path Integrity")
        axs[1, 1].legend(loc="center left")


if __name__ == "__main__":
    G = make_random_network(100, 0.2)
    source = 0
    target = 99
    init_path = next(nx.all_simple_paths(G, source, target))
    # init_path = nx.shortest_path(G, source, target, weight="latency")

    annealer = PathAnnealer(init_path, G, factors={"latency": 0.05, "throughput": -10, "integrity": -10})
    print("Init path:", init_path)
    print("Init energy:", annealer.energy())
    print("Init path length:", len(init_path))
    print("Init path latency:", single_metric_path_cost(G, init_path, "latency"))
    print("Init path throughput:", single_metric_path_cost(G, init_path, "throughput"))
    print("Init path integrity belief:", single_metric_path_cost(G, init_path, "integrity").belief)
    print("Init path integrity uncertainty:", single_metric_path_cost(G, init_path, "integrity").uncertainty)

    annealer.steps = 6000
    annealer.Tmax = 10.0
    annealer.Tmin = 0.000002

    best_path, best_cost = annealer.anneal()
    print()
    print("Best path:", best_path)
    print("Best energy:", best_cost)

    print("Best path length:", len(best_path))
    print("Best path latency:", single_metric_path_cost(G, best_path, "latency"))
    print("Best path throughput:", single_metric_path_cost(G, best_path, "throughput"))
    print("Best path integrity belief:", single_metric_path_cost(G, best_path, "integrity").belief)
    print("Best path integrity uncertainty:", single_metric_path_cost(G, best_path, "integrity").uncertainty)

    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    annealer.plot(axs)
    fig.suptitle("Path Optimization")
    fig.tight_layout()

    # fig.savefig("annealer_plot.png", bbox_inches="tight", dpi=300, transparent=False)
    plt.show()
