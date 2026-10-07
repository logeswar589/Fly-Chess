import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import time
import chess
import numpy as np
import pygame as pg
import pytest
import torch

from fly_chess.core.actions import encode_move
from fly_chess.core.rules import ChessGame
from fly_chess.game.session import HumanSession
from fly_chess.game.opponent import OpponentSnapshot, observe, rebuild_profile
from fly_chess.neural.inference import Evaluator, position_id
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything
from fly_chess.neural.weights import save_weights
from fly_chess.search.mcts import MCTS, SearchSettings
from fly_chess.ui.app import Application
from fly_chess.ui.board import square_at, square_rect
from fly_chess.ui.worker import BrainWorker


@pytest.fixture
def evaluator():
    seed_everything(71)
    return Evaluator(PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=1)), model_id='test')


@pytest.fixture
def app(tmp_path):
    pg.init()
    instance = Application(tmp_path)
    yield instance
    instance.close()
    pg.quit()


def test_session_rejects_stale_ai_after_undo_and_new_match():
    session = HumanSession()
    session.move_human('e2e4')
    old = session.token
    assert session.undo()
    session.move_human('d2d4')
    assert not session.apply_ai(old, 'e7e5')
    assert session.apply_ai(session.token, 'd7d5')
    assert session.undo()
    assert session.game.board.fen() == chess.STARTING_FEN
    assert not HumanSession().apply_ai(session.token, 'e7e5')


def test_black_undo_preserves_initial_ai_move():
    session = HumanSession(chess.BLACK)
    session.apply_ai(session.token, 'e2e4')
    assert not session.can_undo
    session.move_human('e7e5')
    session.apply_ai(session.token, 'g1f3')
    assert session.undo()
    assert session.human_turn
    assert session.history() == ['e4']


def test_human_rules_castle_en_passant_promote_mate_and_draw():
    castle = HumanSession(game=ChessGame.from_fen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'))
    castle.move_human('e1g1')
    assert castle.game.board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    ep = HumanSession(game=ChessGame.from_fen('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'))
    ep.move_human('e5d6')
    assert ep.captured()[chess.WHITE] == [chess.PAWN]
    assert ep.game.board.piece_at(chess.D5) is None
    promotion = HumanSession(game=ChessGame.from_fen('7k/P7/8/8/8/8/8/4K3 w - - 0 1'))
    promotion.move_human('a7a8n')
    assert promotion.game.board.piece_type_at(chess.A8) == chess.KNIGHT
    mate = HumanSession(game=ChessGame.from_fen('7k/8/5KQ1/8/8/8/8/8 w - - 0 1'))
    mate.move_human('g6g7')
    assert mate.finished and mate.game.board.is_checkmate()
    draw = HumanSession(game=ChessGame.from_fen('8/8/8/8/8/5kq1/8/7K w - - 0 1'))
    assert draw.finished and draw.game.board.is_stalemate()
    session = HumanSession()
    for _ in range(2):
        session.move_human('g1f3')
        session.apply_ai(session.token, 'g8f6')
        session.move_human('f3g1')
        session.apply_ai(session.token, 'f6g8')
    assert session.claim_draw()
    assert session.finished
    assert session.undo() and not session.finished
    session.resign()
    assert session.finished and not session.apply_ai(session.token, 'a7a6')


def test_profile_learns_only_human_decisions_rebuilds_and_preserves_weights(evaluator):
    before = {name: value.clone() for name, value in evaluator.model.state_dict().items()}
    game = ChessGame()
    for move in ('e2e4', 'e7e5', 'g1f3', 'b8c6'):
        game.push(move)
    profile = rebuild_profile(game, chess.WHITE, evaluator)
    assert profile.decisions == 2
    assert 0 < profile.strength < .2
    assert profile == rebuild_profile(game, chess.WHITE, evaluator)
    game.undo()
    assert profile == rebuild_profile(game, chess.WHITE, evaluator)
    game.undo()
    assert rebuild_profile(game, chess.WHITE, evaluator).decisions == 1
    assert rebuild_profile(ChessGame(), chess.WHITE, evaluator) == OpponentSnapshot()
    assert all(torch.equal(before[name], value) for name, value in evaluator.model.state_dict().items())


def test_profile_prediction_is_pre_move_normalized_and_forced_moves_do_not_learn():
    game = ChessGame()
    actions = game.actions()
    baseline = np.ones(len(actions))/len(actions)
    original = OpponentSnapshot()
    updated = observe(original, game, encode_move(chess.Move.from_uci('e2e4')), baseline)
    assert updated.log_loss_sum == pytest.approx(np.log(len(actions)))
    assert updated.weights[2] > 0
    prediction = updated.predict(game, actions, baseline)
    assert prediction.sum() == pytest.approx(1.) and np.all(prediction > 0)
    forced = ChessGame.from_fen('7k/8/5K2/8/8/8/7R/8 b - - 0 1')
    assert len(forced.actions()) == 1
    after = observe(updated, forced, int(forced.actions()[0]), np.ones(1))
    assert after.weights == updated.weights and after.informative == updated.informative
    assert original.weights == (0., 0., 0.)


def test_progress_does_not_change_search_and_is_detached(evaluator):
    settings = SearchSettings(simulations=16)
    game = ChessGame()
    snapshots = []
    result = MCTS(evaluator, settings).search(game, on_progress=snapshots.append)
    baseline = MCTS(evaluator, settings).search(game)
    assert result.action == baseline.action
    np.testing.assert_array_equal(result.policy, baseline.policy)
    assert [s['simulations'] for s in snapshots] == [8, 16]
    assert sum(r['visits'] for r in snapshots[0]['root_moves']) == 8
    assert snapshots[0]['position_id'] == position_id(game)


def test_board_coordinates_and_promotion_input(app):
    for flipped in (True, False):
        for square in chess.SQUARES:
            assert square_at(square_rect(square, flipped).center, flipped) == square
    assert square_at((0, 0)) is None
    app.session = HumanSession(game=ChessGame.from_fen('7k/P7/8/8/8/8/8/4K3 w - - 0 1'))
    app.screen = 'play'
    app.board_click(chess.A7)
    app.board_click(chess.A8)
    assert {m.promotion for m in app.promotion} == {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    app.draw()
    app.play_move(next(m for m in app.promotion if m.promotion == chess.ROOK))
    assert app.session.game.board.piece_type_at(chess.A8) == chess.ROOK


def test_freeze_survives_current_updates_and_record_retention(app):
    old = {'ply': 1}
    app.data = old
    app.toggle_freeze()
    app.data = {'ply': 2}
    assert app.visible_data() is old
    app.toggle_freeze()
    assert app.visible_data()['ply'] == 2
    app.decisions = [old]
    app.step_record(-1)
    app.decisions = [{'ply': 3}]
    assert app.visible_data() is old


def test_worker_owns_model_bounds_messages_and_reports_errors(tmp_path, evaluator):
    path = tmp_path / 'tiny.pt'
    save_weights(evaluator.model, path, model_id='ui-fixture')
    before = path.read_bytes()
    worker = BrainWorker()
    game = ChessGame()
    game.push('e2e4')
    try:
        assert worker.submit(token=('match', 1), game=game, human_color=chess.WHITE,
                             weights=path, search=True, adapt=True, difficulty=0)
        worker.future.result(timeout=30)
        messages = worker.drain()
        final = messages[-1]
        assert len(messages) <= 4 and final['kind'] == 'result'
        assert final['move'] in game.board.legal_moves
        assert final['profile']['decisions'] == 1
        assert final['evaluation'].inspection['fen'] == game.board.fen()
        assert path.read_bytes() == before
        assert worker.submit(token=('match', 2), game=game, human_color=True, weights=tmp_path/'missing.pt')
        worker.future.result(timeout=30)
        assert worker.drain()[-1]['kind'] == 'error'
    finally:
        worker.close()


def test_app_remains_interactive_and_rejects_old_messages(app):
    app.new_game()
    app.play_move(chess.Move.from_uci('e2e4'))
    old = app.session.token
    app.undo()
    app.worker.publish({'token': old, 'kind': 'result', 'move': chess.Move.from_uci('e7e5')})
    app.screen = 'settings'  # Do not schedule a new worker for this stale-message check.
    app.tick()
    assert app.session.game.board.fen() == chess.STARTING_FEN
    for screen in ('menu', 'train', 'settings', 'play'):
        app.screen = screen
        surface = app.draw()
        assert surface.get_size() == (1440, 960)
        app.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=0))
        assert app.focus >= 0
