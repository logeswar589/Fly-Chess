"""Absolute-square action encoding, schema v1 (standard chess only)."""

from numbers import Integral
import chess
import numpy as np

ACTION_SCHEMA_VERSION = 1
ACTION_SIZE = 64 * 64 * 5
PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def encode_move(move: chess.Move) -> int:
    if (not move or move.drop is not None or move.promotion not in PROMOTIONS
            or not 0 <= move.from_square < 64 or not 0 <= move.to_square < 64
            or move.from_square == move.to_square):
        raise ValueError(f"Unsupported standard-chess move: {move}")
    return (move.from_square * 64 + move.to_square) * 5 + PROMOTIONS.index(move.promotion)


def decode_action(action: int) -> chess.Move:
    """Decode an action; board-specific legality is checked separately."""
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action must be an integer in [0, {ACTION_SIZE})")
    squares, promotion = divmod(int(action), 5)
    source, destination = divmod(squares, 64)
    if source == destination:
        raise ValueError("Action is an unused same-square slot")
    return chess.Move(source, destination, promotion=PROMOTIONS[promotion])


def legal_action_ids(board: chess.Board, *, claim_draws: bool = False) -> np.ndarray:
    if board.chess960 or type(board) is not chess.Board:
        raise ValueError("Only standard chess is supported")
    if board.is_game_over(claim_draw=claim_draws):
        return np.empty(0, dtype=np.int64)
    return np.asarray([encode_move(move) for move in board.legal_moves], dtype=np.int64)


def legal_mask(board: chess.Board, *, claim_draws: bool = False) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    mask[legal_action_ids(board, claim_draws=claim_draws)] = True
    return mask
