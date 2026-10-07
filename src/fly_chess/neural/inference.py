from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from fly_chess.core.actions import ACTION_SIZE, decode_action, legal_mask
from fly_chess.core.encoding import encode_board
from fly_chess.core.rules import ChessGame
from fly_chess.neural.inspection import ActivationCapture, InspectionRequest


def masked_policy(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Terminal rows are zero; corrupt legal logits fail rather than invent a move."""
    if logits.ndim != 2 or logits.shape[1] != ACTION_SIZE or logits.shape != masks.shape:
        raise ValueError(f"Expected matching [batch, {ACTION_SIZE}] logits and masks")
    if masks.dtype != torch.bool or not logits.is_floating_point() or masks.device != logits.device:
        raise ValueError("Masks must be boolean and on the logits' device")
    if not torch.isfinite(logits[masks]).all():
        raise ValueError("Non-finite legal policy logits")
    active = masks.any(dim=1)
    probabilities = torch.zeros_like(logits)
    probabilities[active] = torch.softmax(logits[active].masked_fill(~masks[active], -torch.inf), dim=1)
    return probabilities


def position_id(game: ChessGame) -> str:
    history = " ".join(move.uci() for move in game.board.move_stack)
    source = f"{game.board.root().fen()}|{history}|{game.board.fen()}|{game.claim_draws}"
    return hashlib.sha256(source.encode()).hexdigest()


@dataclass(frozen=True)
class Evaluation:
    policy: np.ndarray
    value: float
    terminal: bool
    model_id: str
    position_id: str
    inspection: dict | None = None


class Evaluator:
    """Own a frozen inference snapshot; never share this model with an optimizer."""
    def __init__(self, model, *, model_id: str = "untrained"):
        self.model = model.eval()
        self.model_id = model_id
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def evaluate(self, games: list[ChessGame], *, inspection: InspectionRequest | None = None) -> list[Evaluation]:
        if inspection is not None and len(games) != 1:
            raise ValueError("Inspect one position at a time to keep telemetry unambiguous")
        if self.model.training:
            raise RuntimeError("Inference model was switched to training mode; use a separate model")
        results = [game.result for game in games]
        active = [i for i, result in enumerate(results) if result is None]
        output: list[Evaluation | None] = [None] * len(games)
        for i, result in enumerate(results):
            if result is not None:
                output[i] = Evaluation(np.zeros(ACTION_SIZE, dtype=np.float32),
                    result.for_player(games[i].board.turn), True, self.model_id, position_id(games[i]))
        if active:
            states = torch.from_numpy(np.stack([encode_board(games[i].board) for i in active])).to(self.device)
            masks = torch.from_numpy(np.stack([
                legal_mask(games[i].board, claim_draws=games[i].claim_draws) for i in active
            ])).to(self.device)
            if not masks.any(dim=1).all():
                raise ValueError("Nonterminal position has no legal actions")
            context = nullcontext()
            if inspection is not None:
                game = games[0]
                context = ActivationCapture(self.model, inspection, {
                    "position_id": position_id(game), "fen": game.board.fen(),
                    "model_id": self.model_id, "side_to_move": "white" if game.board.turn else "black",
                    "value_perspective": "player_to_move",
                })
            with context as capture:
                logits, values = self.model(states)
            if values.shape != (len(active),) or not torch.isfinite(values).all() or (values.abs() > 1).any():
                raise ValueError("Network returned invalid values; expected finite [batch] values in [-1, 1]")
            policies = masked_policy(logits, masks).cpu().numpy()
            values = values.cpu().numpy()
            for row, i in enumerate(active):
                output[i] = Evaluation(policies[row].copy(), float(values[row]), False,
                    self.model_id, position_id(games[i]), capture.snapshot if capture else None)
        return output


class PolicyAgent:
    """Policy-only experimental agent; no MCTS improvement or strength claim."""
    def __init__(self, evaluator: Evaluator, *, seed: int = 42):
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)

    def choose_move(self, game: ChessGame, *, temperature: float = 0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        result = self.evaluator.evaluate([game])[0]
        if result.terminal:
            return None
        actions = game.actions()
        probabilities = result.policy[actions].astype(np.float64)
        if temperature == 0:
            action = int(actions[np.argmax(probabilities)])
        else:
            with np.errstate(over="ignore", divide="ignore"):
                log_probabilities = np.log(probabilities)
                scaled = (log_probabilities - log_probabilities.max()) / temperature
            weights = np.exp(scaled)
            weights /= weights.sum()
            action = int(self.rng.choice(actions, p=weights))
        move = decode_action(action)
        if move not in game.board.legal_moves:
            raise RuntimeError("Policy agent produced an illegal move")
        return move
