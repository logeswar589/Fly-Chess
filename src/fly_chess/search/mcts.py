from collections import deque
from dataclasses import dataclass, field
import math
from time import monotonic
from typing import Callable, Protocol

import numpy as np

from fly_chess.core.actions import ACTION_SIZE, decode_action
from fly_chess.core.rules import ChessGame
from fly_chess.neural.inference import Evaluation, position_id


class PositionEvaluator(Protocol):
    model_id: str

    def evaluate(self, games: list[ChessGame]) -> list[Evaluation]: ...


# The future match profile receives copies and returns probabilities in action order.
OpponentPrior = Callable[[ChessGame, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SearchSettings:
    simulations: int = 32
    cpuct: float = 1.5
    max_seconds: float | None = None
    max_depth: int = 128
    max_nodes: int = 10000
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25
    opponent_weight: float = 0.0
    use_mcts: bool = True

    def __post_init__(self):
        for name in ("simulations", "max_depth", "max_nodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cpuct", "dirichlet_alpha"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, maximum in (("noise_fraction", 1), ("opponent_weight", 0.25)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be in [0, {maximum}]")
        if self.max_seconds is not None and (
            type(self.max_seconds) not in (float, int) or not math.isfinite(self.max_seconds) or self.max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative, or None")
        if type(self.use_mcts) is not bool:
            raise ValueError("use_mcts must be bool")


@dataclass
class Edge:
    action: int
    network_prior: float
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    edges: dict[int, Edge] = field(default_factory=dict)
    expanded: bool = False
    terminal: bool = False
    value: float = 0.0  # Player to move at this node.


def backup(path: list[Edge], leaf_value: float) -> None:
    """Each edge stores the value for its parent player, not the leaf player."""
    for edge in reversed(path):
        leaf_value = -leaf_value
        edge.visits += 1
        edge.value_sum += leaf_value


def select_edge(node: Node, cpuct: float) -> Edge:
    scale = math.sqrt(1 + sum(edge.visits for edge in node.edges.values()))
    return max(node.edges.values(), key=lambda edge: (
        edge.q + cpuct * edge.prior * scale / (1 + edge.visits),
        edge.prior, -edge.action,
    ))


@dataclass(frozen=True)
class SearchResult:
    action: int | None
    policy: np.ndarray  # Un-tempered normalized visits, or explicitly labeled fallback.
    value: float | None
    simulations: int
    stop_reason: str
    policy_source: str
    snapshot: dict

    @property
    def move(self):
        return None if self.action is None else decode_action(self.action)


class MCTS:
    """One worker owns the evaluator/RNG. A fresh tree is used for every call."""
    def __init__(self, evaluator: PositionEvaluator, settings: SearchSettings | None = None,
                 *, seed: int = 42, clock: Callable[[], float] = monotonic):
        self.evaluator = evaluator
        self.settings = settings or SearchSettings()
        self.rng = np.random.default_rng(seed)
        self.clock = clock

    def _expand(self, node: Node, game: ChessGame, root_turn: bool,
                opponent_prior: OpponentPrior | None) -> None:
        result = game.result
        if result is not None:
            node.value = result.for_player(game.board.turn)
            node.terminal = node.expanded = True
            return
        evaluation = self.evaluator.evaluate([game])[0]
        if evaluation.terminal or evaluation.position_id != position_id(game) or evaluation.model_id != self.evaluator.model_id:
            raise ValueError("Evaluator returned a stale or mismatched search result")
        actions = game.actions()
        policy = np.asarray(evaluation.policy, dtype=np.float64)
        if (policy.shape != (ACTION_SIZE,) or not np.isfinite(policy).all() or (policy < 0).any()
                or not math.isfinite(evaluation.value) or abs(evaluation.value) > 1):
            raise ValueError("Invalid evaluator policy or value")
        baseline = policy[actions]
        if not len(actions) or not math.isfinite(float(baseline.sum())) or baseline.sum() <= 0:
            raise ValueError("Nonterminal search position has no legal policy mass")
        baseline = baseline / baseline.sum()
        # A tiny floor preserves exploration even after floating-point underflow.
        prior = np.maximum(baseline, 1e-8)
        prior /= prior.sum()
        if opponent_prior is not None and game.board.turn != root_turn:
            predicted = np.asarray(opponent_prior(game.copy(), actions.copy(), baseline.copy()), dtype=np.float64)
            if (predicted.shape != baseline.shape or not np.isfinite(predicted).all()
                    or (predicted < 0).any() or not math.isfinite(float(predicted.sum())) or predicted.sum() <= 0):
                raise ValueError("Opponent prior must provide finite legal probabilities")
            weight = self.settings.opponent_weight
            prior = (1 - weight) * prior + weight * predicted / predicted.sum()
        node.edges = {int(action): Edge(int(action), float(base), float(p))
                      for action, base, p in zip(actions, baseline, prior)}
        node.value = float(evaluation.value)
        node.expanded = True

    def search(self, game: ChessGame, *, temperature: float = 0, self_play: bool = False,
               cancel: Callable[[], bool] | None = None, request_id: str = "search",
               game_id: str = "unspecified", tree_edges: int = 0,
               opponent_prior: OpponentPrior | None = None,
               on_progress: Callable[[dict], None] | None = None) -> SearchResult:
        if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(tree_edges) is not int or not 0 <= tree_edges <= 4096:
            raise ValueError("tree_edges must be an integer in [0, 4096]")
        if opponent_prior is not None and (self_play or not self.settings.use_mcts or self.settings.opponent_weight == 0):
            raise ValueError("Opponent prior requires MCTS play mode and a positive bounded weight")
        root_game = game.copy()
        root = Node()
        started = self.clock()
        deadline = None if self.settings.max_seconds is None else started + self.settings.max_seconds
        completed, nodes, depth_hits = 0, 1, 0
        noise_applied = False

        def interrupted():
            if cancel is not None and cancel():
                return "cancelled"
            if deadline is not None and self.clock() >= deadline:
                return "time_limit"
            return None

        reason = interrupted()
        if reason is None:
            self._expand(root, root_game, root_game.board.turn, None)
            reason = interrupted()
        if reason is None and root.terminal:
            reason = "terminal"
        if reason is None and not self.settings.use_mcts:
            reason = "policy_only"
        if reason is None and self_play and self.settings.noise_fraction:
            noise_applied = True
            noise = self.rng.dirichlet(np.full(len(root.edges), self.settings.dirichlet_alpha))
            for edge, sample in zip(root.edges.values(), noise):
                edge.prior = (1 - self.settings.noise_fraction) * edge.prior + self.settings.noise_fraction * float(sample)
        while reason is None and completed < self.settings.simulations:
            current = root_game.copy()
            node, path = root, []
            while node.expanded and not node.terminal and len(path) < self.settings.max_depth:
                reason = interrupted()
                if reason:
                    break
                edge = select_edge(node, self.settings.cpuct)
                if edge.child is None:
                    if nodes >= self.settings.max_nodes:
                        reason = "node_limit"
                        break
                    edge.child = Node()
                    nodes += 1
                current.push_action(edge.action)
                path.append(edge)
                node = edge.child
            if reason:
                break
            reason = interrupted()
            if reason:
                break
            if not node.expanded:
                self._expand(node, current, root_game.board.turn, opponent_prior)
            reason = interrupted()
            if reason:
                break  # In-flight evaluation is not a completed simulation.
            if len(path) == self.settings.max_depth and not node.terminal:
                depth_hits += 1
            backup(path, node.value)
            completed += 1
            if on_progress is not None and completed % 8 == 0:
                progress = self._snapshot(root, min(tree_edges, 128))
                progress.update({"request_id": request_id, "game_id": game_id,
                    "position_id": position_id(root_game), "model_id": self.evaluator.model_id,
                    "fen": root_game.board.fen(), "simulations": completed,
                    "allocated_nodes": nodes, "stop_reason": "searching",
                    "elapsed_seconds": max(0., self.clock() - started)})
                on_progress(progress)
        reason = reason or "simulation_limit"
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        source = "none"
        if root.edges:
            edges = list(root.edges.values())
            weights = np.array([edge.visits for edge in edges], dtype=np.float64)
            if weights.sum():
                source = "mcts_visits"
            else:
                weights = np.array([edge.network_prior for edge in edges])
                source = "network_policy" if reason == "policy_only" else "network_fallback_no_visits"
            weights /= weights.sum()
            policy[[edge.action for edge in edges]] = weights
        action = None
        if policy.any() and reason != "cancelled":
            legal = np.flatnonzero(policy)
            if temperature == 0:
                action = int(legal[np.argmax(policy[legal])])
            else:
                logs = np.log(policy[legal].astype(np.float64))
                with np.errstate(over="ignore", under="ignore"):
                    probabilities = np.exp((logs - logs.max()) / temperature)
                probabilities /= probabilities.sum()
                action = int(self.rng.choice(legal, p=probabilities))
        value = (sum(edge.value_sum for edge in root.edges.values()) / completed
                 if completed else (root.value if root.expanded else None))
        snapshot = self._snapshot(root, tree_edges)
        snapshot.update({
            "request_id": request_id, "game_id": game_id, "position_id": position_id(root_game),
            "model_id": self.evaluator.model_id, "fen": root_game.board.fen(),
            "side_to_move": "white" if root_game.board.turn else "black",
            "value_perspective": "root player for root_value; parent player for each edge q",
            "simulations": completed, "allocated_nodes": nodes, "depth_cutoffs": depth_hits,
            "elapsed_seconds": max(0.0, self.clock() - started), "stop_reason": reason,
            "policy_source": source, "root_value": value,
            "network_value": root.value if root.expanded and not root.terminal else None,
            "selected_move": None if action is None else decode_action(action).uci(),
            "self_play_noise": noise_applied,
            "opponent_prior_enabled": opponent_prior is not None,
        })
        return SearchResult(action, policy, value, completed, reason, source, snapshot)

    @staticmethod
    def _snapshot(root: Node, limit: int) -> dict:
        def row(edge):
            return {"move": decode_action(edge.action).uci(), "action": edge.action,
                    "visits": edge.visits, "q": edge.q, "network_prior": edge.network_prior,
                    "search_prior": edge.prior}
        ordered = lambda node: sorted(node.edges.values(), key=lambda e: (-e.visits, -e.prior, e.action))
        principal, node = [], root
        for _ in range(16):
            if not node.edges:
                break
            edge = ordered(node)[0]
            if not edge.visits:
                break
            principal.append(decode_action(edge.action).uci())
            if edge.child is None:
                break
            node = edge.child
        tree, queue = [], deque([(root, [])])
        while queue and len(tree) < limit:
            node, path = queue.popleft()
            for edge in ordered(node):
                if len(tree) == limit:
                    break
                tree.append({"path": path, **row(edge)})
                if edge.child is not None:
                    queue.append((edge.child, [*path, decode_action(edge.action).uci()]))
        return {"root_moves": [row(edge) for edge in ordered(root)], "principal_variation": principal,
                "tree": tree, "tree_edge_limit": limit}
