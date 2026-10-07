"""Validated game boundary: no unchecked board.push calls outside this adapter."""

from dataclasses import dataclass
import chess

from fly_chess.core.actions import decode_action, legal_action_ids


@dataclass(frozen=True)
class GameResult:
    white_value: float
    termination: str

    def for_player(self, color: chess.Color) -> float:
        return self.white_value if color == chess.WHITE else -self.white_value


def outcome(board: chess.Board, *, claim_draws: bool = False) -> GameResult | None:
    result = board.outcome(claim_draw=claim_draws)
    if result is None:
        return None
    value = 0.0 if result.winner is None else (1.0 if result.winner else -1.0)
    return GameResult(value, result.termination.name.lower())


class ChessGame:
    def __init__(self, board: chess.Board | None = None, *, claim_draws: bool = False):
        self.board = chess.Board() if board is None else board.copy(stack=True)
        if type(self.board) is not chess.Board or self.board.chess960:
            raise ValueError("Only standard chess is supported")
        if not self.board.is_valid():
            raise ValueError(f"Invalid chess position (status={self.board.status()})")
        self.claim_draws = claim_draws

    @classmethod
    def from_fen(cls, fen: str, *, claim_draws: bool = False) -> "ChessGame":
        # A FEN does not carry repetition history. Use copy() for search.
        return cls(chess.Board(fen), claim_draws=claim_draws)

    def copy(self) -> "ChessGame":
        return ChessGame(self.board, claim_draws=self.claim_draws)

    @property
    def result(self) -> GameResult | None:
        return outcome(self.board, claim_draws=self.claim_draws)

    def actions(self):
        return legal_action_ids(self.board, claim_draws=self.claim_draws)

    def push(self, move: chess.Move | str) -> None:
        if self.result is not None:
            raise ValueError("Cannot move after the game has ended")
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        self.board.push(move)

    def push_action(self, action: int) -> None:
        self.push(decode_action(action))

    def undo(self) -> chess.Move:
        if not self.board.move_stack:
            raise ValueError("No moves to undo")
        return self.board.pop()
