import json

import chess
import numpy as np
import pytest

from fly_chess.core.actions import ACTION_SIZE, decode_action, encode_move
from fly_chess.core.rules import ChessGame
from fly_chess.neural.inference import Evaluation, Evaluator, position_id
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything
from fly_chess.search.mcts import Edge, MCTS, Node, SearchSettings, backup, select_edge


class StubEvaluator:
    model_id = "test-fixture"

    def __init__(self, value=None, priorities=None):
        self.value = value or (lambda game: 0.)
        self.priorities = priorities
        self.calls = 0

    def evaluate(self, games):
        self.calls += 1
        output = []
        for game in games:
            assert game.result is None, "Search must resolve terminals without the network"
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            actions = game.actions()
            policy[actions] = 1
            if self.priorities:
                priorities = self.priorities(game)
                policy[actions] = 1e-8
                for move, weight in priorities.items():
                    action = encode_move(chess.Move.from_uci(move))
                    if action in actions:
                        policy[action] = weight
            policy /= policy.sum()
            output.append(Evaluation(policy, self.value(game), False, self.model_id, position_id(game)))
        return output


def test_backup_changes_perspective_at_every_ply():
    path = [Edge(i, 1, 1) for i in range(3)]
    backup(path, 0.75)
    assert [edge.q for edge in path] == [-0.75, 0.75, -0.75]
    assert [edge.visits for edge in path] == [1, 1, 1]
    backup(path, -0.25)
    assert [edge.q for edge in path] == [-0.25, 0.25, -0.25]
    node = Node(edges={i: edge for i, edge in enumerate(path)})
    assert select_edge(node, 1).action == 1


@pytest.mark.parametrize("fen", [
    "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/5kq1/8/7K b - - 0 1",
])
def test_search_finds_mate_for_either_color(fen):
    game = ChessGame.from_fen(fen)
    before = game.board.fen()
    result = MCTS(StubEvaluator(), SearchSettings(simulations=128)).search(game)
    assert result.move in game.board.legal_moves
    child = game.copy()
    child.push(result.move)
    assert child.board.is_checkmate()
    selected = next(row for row in result.snapshot["root_moves"] if row["action"] == result.action)
    assert selected["q"] == 1
    assert game.board.fen() == before and not game.board.move_stack


def test_opponent_chooses_adversarial_reply_not_cooperative_reply():
    def value(game):
        moves = [move.uci() for move in game.board.move_stack]
        white_value = 0.1
        if moves and moves[0] == "d2d4":
            white_value = -0.9 if len(moves) > 1 and moves[1] == "g8f6" else 0.9
        return white_value if game.board.turn else -white_value
    def priors(game):
        if not game.board.move_stack:
            return {"d2d4": 0.5, "e2e4": 0.5}
        return {"g8f6": 0.5, "d7d5": 0.5}
    result = MCTS(StubEvaluator(value, priors), SearchSettings(simulations=128, max_depth=4)).search(ChessGame())
    assert result.move.uci() == "e2e4"
    rows = {row["move"]: row for row in result.snapshot["root_moves"]}
    assert rows["d2d4"]["q"] < 0 < rows["e2e4"]["q"]


def test_visit_accounting_normalization_and_bounded_snapshot():
    game = ChessGame()
    result = MCTS(StubEvaluator(), SearchSettings(simulations=40, max_depth=2)).search(
        game, tree_edges=7, request_id="req", game_id="game")
    assert result.simulations == sum(row["visits"] for row in result.snapshot["root_moves"]) == 40
    assert result.policy.sum() == pytest.approx(1)
    assert set(np.flatnonzero(result.policy)) <= set(game.actions())
    assert len(result.snapshot["tree"]) == 7
    assert result.snapshot["depth_cutoffs"] > 0
    assert result.snapshot["request_id"] == "req" and result.snapshot["game_id"] == "game"
    line = game.copy()
    for move in result.snapshot["principal_variation"]:
        line.push(move)
    json.dumps(result.snapshot, allow_nan=False)


def test_repetition_and_terminal_bypass_preserve_history():
    game = ChessGame(claim_draws=True)
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        # Claim policy would end on the intended final move; build the fixture via rules library.
        game.board.push_uci(move)
    evaluator = StubEvaluator()
    result = MCTS(evaluator).search(game)
    assert result.stop_reason == "terminal" and result.move is None and result.value == 0
    assert result.policy.sum() == 0 and evaluator.calls == 0
    assert len(game.board.move_stack) == 8
    # Same FEN, without history, is not terminal.
    assert MCTS(evaluator, SearchSettings(simulations=1)).search(ChessGame.from_fen(game.board.fen())).move is not None


def test_repetition_draw_is_found_below_root():
    game = ChessGame(claim_draws=True)
    for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        game.push(move)
    assert game.result is None
    evaluator = StubEvaluator(priorities=lambda game: {"f3g1": 1.})
    result = MCTS(evaluator, SearchSettings(simulations=1)).search(game)
    assert result.move.uci() == "f3g1" and result.value == 0
    assert evaluator.calls == 1
    child = game.copy()
    child.push(result.move)
    assert child.result.termination == "threefold_repetition"


def test_bad_evaluator_and_opponent_predictions_fail_explicitly():
    class Stale(StubEvaluator):
        def evaluate(self, games):
            result = super().evaluate(games)[0]
            return [Evaluation(result.policy, result.value, False, self.model_id, "old-position")]
    with pytest.raises(ValueError, match="stale"):
        MCTS(Stale()).search(ChessGame())
    with pytest.raises(ValueError, match="Invalid evaluator"):
        MCTS(StubEvaluator(value=lambda game: float("nan"))).search(ChessGame())
    with pytest.raises(ValueError, match="Opponent prior"):
        MCTS(StubEvaluator(), SearchSettings(opponent_weight=0.2)).search(
            ChessGame(), opponent_prior=lambda *args: np.array([float("nan")]))


def test_cancellation_before_and_during_search():
    evaluator = StubEvaluator()
    result = MCTS(evaluator).search(ChessGame(), cancel=lambda: True)
    assert result.stop_reason == "cancelled" and result.action is None and evaluator.calls == 0
    result = MCTS(evaluator).search(ChessGame(), cancel=lambda: evaluator.calls >= 3)
    assert result.stop_reason == "cancelled" and result.action is None
    assert result.simulations == 1  # Last in-flight leaf result was discarded.
    assert sum(row["visits"] for row in result.snapshot["root_moves"]) == 1


def test_time_and_node_budgets():
    evaluator = StubEvaluator()
    result = MCTS(evaluator, SearchSettings(max_seconds=0)).search(ChessGame())
    assert result.stop_reason == "time_limit" and result.move is None and evaluator.calls == 0
    time = [0.]
    class SlowEvaluator(StubEvaluator):
        def evaluate(self, games):
            output = super().evaluate(games)
            time[0] += 2
            return output
    result = MCTS(SlowEvaluator(), SearchSettings(max_seconds=1), clock=lambda: time[0]).search(ChessGame())
    assert result.stop_reason == "time_limit" and result.simulations == 0
    assert result.policy_source == "network_fallback_no_visits" and result.move is not None
    result = MCTS(evaluator, SearchSettings(max_nodes=1)).search(ChessGame())
    assert result.stop_reason == "node_limit" and result.snapshot["allocated_nodes"] == 1


def test_noise_is_selfplay_only_and_seeded_and_temperature_leaves_targets_unchanged():
    def run(seed, self_play=False, temperature=0):
        return MCTS(StubEvaluator(), SearchSettings(simulations=10), seed=seed).search(
            ChessGame(), self_play=self_play, temperature=temperature)
    normal = run(1)
    assert normal.snapshot["root_moves"] == run(999).snapshot["root_moves"]
    noisy = run(1, True)
    assert noisy.snapshot["root_moves"] == run(1, True).snapshot["root_moves"]
    assert noisy.snapshot["root_moves"] != run(2, True).snapshot["root_moves"]
    np.testing.assert_array_equal(normal.policy, run(1, temperature=0.7).policy)


def test_policy_only_is_explicit_and_does_not_build_tree():
    evaluator = StubEvaluator()
    result = MCTS(evaluator, SearchSettings(use_mcts=False)).search(ChessGame())
    assert result.stop_reason == "policy_only" and result.policy_source == "network_policy"
    assert result.simulations == 0 and evaluator.calls == 1


def test_opponent_hook_only_changes_opponent_exploration_with_bounded_influence():
    turns = []
    def opponent(game, actions, baseline):
        turns.append(game.board.turn)
        prediction = np.zeros_like(baseline)
        prediction[0] = 1
        return prediction
    result = MCTS(StubEvaluator(), SearchSettings(simulations=25, opponent_weight=0.2)).search(
        ChessGame(), opponent_prior=opponent, tree_edges=100)
    assert turns and not any(turns)
    for row in result.snapshot["root_moves"]:
        assert row["search_prior"] == pytest.approx(row["network_prior"])
    assert all(row["search_prior"] > 0 for row in result.snapshot["tree"])
    with pytest.raises(ValueError, match="play mode"):
        MCTS(StubEvaluator()).search(ChessGame(), self_play=True, opponent_prior=opponent)


def test_real_network_search_reproduces_on_cpu():
    seed_everything(8, cpu_threads=1)
    evaluator = Evaluator(PolicyValueNetwork(NetworkSpec(8, 1)))
    first = MCTS(evaluator, SearchSettings(simulations=8), seed=5).search(ChessGame(), self_play=True)
    second = MCTS(evaluator, SearchSettings(simulations=8), seed=5).search(ChessGame(), self_play=True)
    np.testing.assert_array_equal(first.policy, second.policy)
    assert first.action == second.action and first.value == second.value


@pytest.mark.parametrize("kwargs", [
    {"simulations": 0}, {"max_seconds": -1}, {"cpuct": float("nan")},
    {"opponent_weight": 0.3}, {"max_nodes": False}, {"use_mcts": "yes"},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SearchSettings(**kwargs)
