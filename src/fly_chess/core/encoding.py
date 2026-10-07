"""21 float32 planes in absolute rank/file orientation; see docs/contracts.md."""

import chess
import numpy as np

ENCODING_SCHEMA_VERSION = 1
PLANE_NAMES = (
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
    "white_to_move", "white_castle_kingside", "white_castle_queenside",
    "black_castle_kingside", "black_castle_queenside", "en_passant",
    "halfmove_clock", "repeated_twice", "repeated_three_times",
)
INPUT_PLANES = len(PLANE_NAMES)


def encode_board(board: chess.Board) -> np.ndarray:
    if board.chess960 or type(board) is not chess.Board:
        raise ValueError("Only standard chess is supported")
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        index = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        planes[index, chess.square_rank(square), chess.square_file(square)] = 1
    planes[12].fill(float(board.turn))
    for index, (color, kingside) in enumerate(
        ((chess.WHITE, True), (chess.WHITE, False), (chess.BLACK, True), (chess.BLACK, False)), 13
    ):
        right = board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)
        planes[index].fill(float(right))
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    planes[18].fill(min(board.halfmove_clock, 150) / 150)
    planes[19].fill(float(board.is_repetition(2)))
    planes[20].fill(float(board.is_repetition(3)))
    return planes
