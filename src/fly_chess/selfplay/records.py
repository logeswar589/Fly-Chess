from dataclasses import dataclass, field
import math

import chess
import numpy as np

from fly_chess.core.actions import ACTION_SCHEMA_VERSION, decode_action
from fly_chess.core.encoding import ENCODING_SCHEMA_VERSION, INPUT_PLANES, encode_board
from fly_chess.core.rules import ChessGame

DATA_SCHEMA_VERSION = 1
SCHEMAS = {"data_schema": DATA_SCHEMA_VERSION, "action_schema": ACTION_SCHEMA_VERSION,
           "encoding_schema": ENCODING_SCHEMA_VERSION}
POLICY_SOURCES = {"mcts_visits", "network_policy", "network_fallback_no_visits"}


@dataclass
class Sample:
    state: np.ndarray
    action: int
    legal_actions: np.ndarray
    policy: np.ndarray  # Sparse probabilities aligned with ALL legal_actions, including zeros.
    white_to_move: bool
    value_prediction: float
    policy_source: str
    target: float | None = None


@dataclass
class GameRecord:
    game_id: str
    seed: int
    model_id: str
    root_fen: str
    prefix_moves: list[str]
    claim_draws: bool
    samples: list[Sample] = field(default_factory=list)
    status: str = "aborted"
    termination: str = "cancelled"
    white_result: float | None = None
    settings: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        return {**SCHEMAS, **{name: getattr(self, name) for name in (
            "game_id", "seed", "model_id", "root_fen", "prefix_moves", "claim_draws",
            "status", "termination", "white_result", "settings",
        )}}

    def validate(self) -> None:
        if not self.game_id or not self.model_id or type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid game identity or seed")
        if type(self.claim_draws) is not bool or self.status not in {"completed", "truncated", "aborted"}:
            raise ValueError("Invalid game status or draw policy")
        game = ChessGame.from_fen(self.root_fen)  # Rebuild history before enabling claim adjudication.
        for move in self.prefix_moves:
            game.push(move)
        game.claim_draws = self.claim_draws
        for sample in self.samples:
            if game.result is not None:
                raise ValueError("Samples continue after terminal position")
            actions = game.actions()
            if (sample.state.dtype != np.float32 or sample.state.shape != (INPUT_PLANES, 8, 8)
                    or not np.array_equal(sample.state, encode_board(game.board))):
                raise ValueError("Stored state does not match game history")
            if sample.legal_actions.dtype.kind not in "iu" or not np.array_equal(sample.legal_actions, actions):
                raise ValueError("Stored legal actions do not match chess rules")
            if (sample.policy.shape != actions.shape or not np.isfinite(sample.policy).all()
                    or (sample.policy < 0).any() or not np.isclose(sample.policy.sum(), 1, atol=1e-6)):
                raise ValueError("Invalid sparse policy")
            if sample.action not in actions or sample.policy_source not in POLICY_SOURCES:
                raise ValueError("Invalid selected action or policy provenance")
            if type(sample.white_to_move) is not bool or sample.white_to_move != game.board.turn:
                raise ValueError("Wrong sample player perspective")
            if not math.isfinite(sample.value_prediction) or abs(sample.value_prediction) > 1:
                raise ValueError("Invalid network value prediction")
            expected = None if self.white_result is None else (self.white_result if sample.white_to_move else -self.white_result)
            if sample.target != expected:
                raise ValueError("Wrong outcome target perspective")
            game.push(decode_action(sample.action))
        if self.status == "completed":
            if game.result is None or self.white_result != game.result.white_value or self.termination != game.result.termination:
                raise ValueError("Completed game outcome does not match chess rules")
        elif self.status == "truncated":
            if game.result is not None or self.termination != "ply_limit" or self.white_result != 0:
                raise ValueError("Invalid truncation convention")
        elif self.white_result is not None or any(s.target is not None for s in self.samples):
            raise ValueError("Aborted games must not have outcome targets")
