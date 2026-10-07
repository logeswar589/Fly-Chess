import random

import chess
import numpy as np
import pytest

from fly_chess.core.actions import ACTION_SIZE, decode_action, encode_move, legal_mask
from fly_chess.core.encoding import INPUT_PLANES, encode_board
from fly_chess.core.rules import ChessGame, outcome


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "7k/P7/8/8/8/8/7p/K7 w - - 0 1",
    "7k/P7/8/8/8/8/7p/K7 b - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    # En passant would expose the king to a rook: it must be masked out.
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
])
def test_legal_action_bijection(fen):
    game = ChessGame.from_fen(fen)
    expected = set(game.board.legal_moves)
    actions = game.actions()
    assert len(set(actions)) == len(expected)
    assert {decode_action(action) for action in actions} == expected
    assert set(np.flatnonzero(legal_mask(game.board))) == set(actions)
    for move in expected:
        assert decode_action(encode_move(move)) == move
        child = game.copy()
        child.push_action(encode_move(move))
        assert child.board.is_valid()


def test_all_promotion_choices_and_castling_present():
    board = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
    assert {move.promotion for move in board.legal_moves if move.promotion} == {2, 3, 4, 5}
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {move.uci() for move in board.legal_moves if board.is_castling(move)} == {"e1g1", "e1c1"}


def test_special_move_effects_and_undo():
    game = ChessGame.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    before = game.board.fen()
    game.push("e1g1")
    assert game.board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert game.board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert not game.board.has_castling_rights(chess.WHITE)
    game.undo()
    assert game.board.fen() == before
    game = ChessGame.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    game.push("e5d6")
    assert game.board.piece_at(chess.D5) is None
    assert game.board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    pinned = ChessGame.from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert not legal_mask(pinned.board)[encode_move(chess.Move.from_uci("e5d6"))]
    for suffix, piece in (("n", chess.KNIGHT), ("b", chess.BISHOP), ("r", chess.ROOK), ("q", chess.QUEEN)):
        game = ChessGame.from_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        game.push(f"a7a8{suffix}")
        assert game.board.piece_at(chess.A8) == chess.Piece(piece, chess.WHITE)


def test_seeded_playouts_never_mutate_parent_or_allow_illegal_actions():
    rng = random.Random(17)
    for _ in range(4):
        game = ChessGame()
        for _ in range(100):
            if game.result:
                break
            actions = game.actions()
            assert len(actions) == game.board.legal_moves.count()
            move = decode_action(rng.choice(actions))
            assert move in game.board.legal_moves
            before = game.board.fen()
            clone = game.copy()
            clone.push(move)
            assert game.board.fen() == before
            game.push(move)
            assert game.board.is_valid()


def test_plane_orientation_and_piece_counts():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (INPUT_PLANES, 8, 8) == (21, 8, 8)
    assert planes.dtype == np.float32
    assert planes[:12].sum() == 32
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color else 6)
        assert planes[plane, square // 8, square % 8] == 1
    assert np.all(planes[12:17] == 1)
    board.push_uci("e2e4")
    after = encode_board(board)
    assert not after[12].any()
    assert after[0, 3, 4] == 1 and after[0, 1, 4] == 0
    assert after[17, 2, 4] == 1 and after[17].sum() == 1


def test_castling_planes_and_clock():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 75 1")
    planes = encode_board(board)
    assert [planes[i, 0, 0] for i in range(13, 17)] == [1, 0, 0, 1]
    assert np.all(planes[18] == 0.5)
    board.halfmove_clock = 200
    assert np.all(encode_board(board)[18] == 1)


def repeated_board(cycles):
    board = chess.Board()
    for _ in range(cycles):
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
    return board


def test_repetition_history_copy_and_claim_policy():
    board = repeated_board(1)
    assert np.all(encode_board(board)[19] == 1)
    assert not encode_board(board)[20].any()
    board = repeated_board(2)
    game = ChessGame(board, claim_draws=True)
    assert game.result.termination == "threefold_repetition"
    assert game.copy().result == game.result
    assert len(game.copy().board.move_stack) == 8
    assert outcome(board, claim_draws=False) is None
    assert not game.actions().size
    assert not legal_mask(board, claim_draws=True).any()
    assert legal_mask(board, claim_draws=False).any()
    assert np.all(encode_board(board)[19:21] == 1)
    # FEN restores the board, not the historical moves needed for repetition.
    assert not encode_board(chess.Board(board.fen()))[19:21].any()
    with pytest.raises(ValueError, match="ended"):
        game.push("e2e4")


@pytest.mark.parametrize("fen,reason,white_value", [
    ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "checkmate", 1),
    ("8/8/8/8/8/6k1/6q1/7K w - - 0 1", "checkmate", -1),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "stalemate", 0),
    ("7k/8/6K1/8/8/8/8/8 w - - 0 1", "insufficient_material", 0),
    ("7k/8/6K1/8/8/8/8/R7 w - - 150 1", "seventyfive_moves", 0),
])
def test_terminal_results_and_empty_masks(fen, reason, white_value):
    game = ChessGame.from_fen(fen)
    assert game.result.termination == reason
    assert game.result.for_player(chess.WHITE) == white_value
    assert game.result.for_player(chess.BLACK) == -white_value
    assert not game.actions().size
    assert not legal_mask(game.board).any()


def test_fifty_move_claim_and_automatic_fivefold():
    board = chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 100 1")
    assert outcome(board) is None
    assert outcome(board, claim_draws=True).termination == "fifty_moves"
    assert outcome(repeated_board(4)).termination == "fivefold_repetition"


@pytest.mark.parametrize("action", [-1, ACTION_SIZE, 1.5, True, 0])
def test_invalid_action_slots(action):
    with pytest.raises(ValueError):
        decode_action(action)


def test_reject_illegal_move_invalid_board_and_undo():
    game = ChessGame()
    before = game.board.fen()
    with pytest.raises(ValueError, match="Illegal"):
        game.push("e2e5")
    assert game.board.fen() == before
    with pytest.raises(ValueError, match="No moves"):
        game.undo()
    game.push("e2e4")
    assert game.undo().uci() == "e2e4"
    assert game.board.fen() == before
    with pytest.raises(ValueError, match="Invalid"):
        ChessGame.from_fen("8/8/8/8/8/8/8/8 w - - 0 1")
    with pytest.raises(ValueError, match="standard"):
        ChessGame(chess.Board.from_chess960_pos(0))
    with pytest.raises(ValueError):
        encode_move(chess.Move.null())
