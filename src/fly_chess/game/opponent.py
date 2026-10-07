"""Regularized online move-choice model; never modifies neural weights."""

from dataclasses import dataclass
import math

import chess
import numpy as np

from fly_chess.core.actions import decode_action, encode_move
from fly_chess.core.rules import ChessGame

FEATURE_NAMES = ("Captures", "Checks", "Central destinations")


def features(game, actions):
    board = game.board
    return np.asarray([[float(board.is_capture(move)), float(board.gives_check(move)),
        float(move.to_square in (chess.D4, chess.E4, chess.D5, chess.E5))]
        for move in map(decode_action, actions)], dtype=np.float64)


@dataclass(frozen=True)
class OpponentSnapshot:
    weights: tuple = (0., 0., 0.)
    decisions: int = 0
    informative: int = 0
    log_loss_sum: float = 0.

    @property
    def strength(self):
        return 0.2 * self.informative / (self.informative + 12)

    def predict(self, game, actions, baseline):
        logits = np.log(np.maximum(baseline, 1e-8)) + features(game, actions) @ np.asarray(self.weights)
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def as_dict(self):
        return {"decisions": self.decisions, "informative": self.informative,
                "influence": self.strength, "weights": list(self.weights),
                "features": FEATURE_NAMES, "mean_pre_move_log_loss": self.log_loss_sum / self.decisions if self.decisions else None,
                "evidence": "Insufficient evidence" if self.informative < 12 else "Limited match evidence",
                "status": "Experimental / resets each match"}


def observe(snapshot, game, action, baseline):
    actions = game.actions()
    if action not in actions:
        raise ValueError("Cannot learn from an illegal human action")
    matrix = features(game, actions)
    prediction = snapshot.predict(game, actions, baseline)
    selected = int(np.flatnonzero(actions == action)[0])
    informative = len(actions) > 1 and bool(np.ptp(matrix, axis=0).any())
    weights = np.asarray(snapshot.weights)
    if informative:
        weights = np.clip(weights + 0.15 * (matrix[selected] - prediction @ matrix - 0.2 * weights), -2, 2)
    return OpponentSnapshot(tuple(weights), snapshot.decisions + 1, snapshot.informative + int(informative),
                            snapshot.log_loss_sum - math.log(max(prediction[selected], 1e-12)))


def rebuild_profile(game: ChessGame, human_color, evaluator, cancel=lambda: False):
    history = ChessGame(game.board.root(), claim_draws=False)
    snapshot = OpponentSnapshot()
    for move in game.board.move_stack:
        if cancel():
            return snapshot
        if history.board.turn == human_color:
            actions = history.actions()
            baseline = evaluator.evaluate([history])[0].policy[actions]
            # Prediction is computed from pre-move state before observation updates it.
            snapshot = observe(snapshot, history, encode_move(move), baseline)
        history.push(move)
    return snapshot
