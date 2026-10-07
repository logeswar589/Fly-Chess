import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from pathlib import Path
import time
import chess
import chess.pgn
import pygame as pg
import pytest
import torch

from fly_chess.config import Config
from fly_chess.core.actions import encode_move
from fly_chess.game.session import HumanSession
from fly_chess.game.human_data import save_human_game, load_human_dataset, HumanArchiveWorker
from fly_chess.neural.network import PolicyValueNetwork, NetworkSpec
from fly_chess.neural.weights import save_weights, load_weights
from fly_chess.storage.checkpoints import file_hash
from fly_chess.training.human import train_human_candidate
from fly_chess.ui.app import Application


def mate_session(color=True):
    session = HumanSession(color)
    for move in (('f2f3', 'e7e5', 'g2g4', 'd8h4') if color else ('e2e4', 'f7f6', 'd2d4', 'g7g5', 'd1h5')):
        if session.human_turn:
            session.move_human(move)
        else:
            session.apply_ai(session.token, move)
    assert session.game.board.is_checkmate()
    return session


def archive(root, session, consent=True):
    return save_human_game(root, session.game, game_id=session.game_id,
        human_color=session.human_color, manual_result=session.manual_result,
        model_id='test-checkpoint', checkpoint='test.pt', consent=consent)


@pytest.mark.parametrize('color', [True, False])
def test_complete_pgn_roundtrip_and_only_human_targets(tmp_path, color):
    session = mate_session(color)
    path = archive(tmp_path, session)
    with path.open(encoding='utf-8') as stream:
        record = chess.pgn.read_game(stream)
    assert not record.errors
    assert record.end().board().fen() == session.game.board.fen()
    assert record.headers['Result'] == session.game.board.result()
    dataset = load_human_dataset(tmp_path)
    assert len(dataset.samples) == 2
    assert all(s.white_to_move == color and s.target == -1 for s in dataset.samples)
    assert [s.action for s in dataset.samples] == [encode_move(chess.Move.from_uci(m)) for m in (('f2f3', 'g2g4') if color else ('f7f6', 'g7g5'))]
    assert all(s.policy.sum() == 1 and s.policy[s.legal_actions == s.action] == 1 for s in dataset.samples)
    assert not (tmp_path/'data/replay_buffer').exists()


def test_storage_opt_in_and_incomplete_undo_resignation(tmp_path):
    session = HumanSession()
    session.move_human('e2e4')
    assert archive(tmp_path, session, consent=False) is None
    assert not (tmp_path/'data').exists()
    session.apply_ai(session.token, 'e7e5')
    session.move_human('g1f3')
    session.undo()
    path = archive(tmp_path, session)
    text = path.read_text()
    assert '[Result "*"]' in text and 'Nf3' not in text
    with pytest.raises(ValueError, match='No completed'):
        load_human_dataset(tmp_path)
    session.resign()
    archive(tmp_path, session)
    data = load_human_dataset(tmp_path)
    assert len(data.samples) == 1 and data.samples[0].target == -1
    assert data.sources[0]['termination'] == 'resignation'


def test_invalid_provenance_and_false_results_rejected(tmp_path):
    path = archive(tmp_path, mate_session())
    original = path.read_text()
    path.write_text(original.replace('[FlySaveConsent "true"]', '[FlySaveConsent "false"]'))
    with pytest.raises(ValueError, match='consent'):
        load_human_dataset(tmp_path)
    path.write_text(original.replace('[Result "0-1"]', '[Result "1-0"]'))
    with pytest.raises(ValueError, match='inconsistent'):
        load_human_dataset(tmp_path)


def test_human_candidate_updates_separate_data_and_does_not_bypass_gate(tmp_path):
    archive(tmp_path, mate_session())
    model = PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=1))
    base = tmp_path/'models/fly_best.pt'
    save_weights(model, base, model_id='parent')
    before = base.read_bytes()
    config = Config(device='cpu', batch_size=4, replay_buffer_size=8, evaluation_pairs=1,
                    evaluation_min_pairs=20, evaluation_simulations=2, evaluation_max_plies=4)
    result = train_human_candidate(tmp_path, base, config)
    assert result['status'] == 'complete'
    assert not result['evaluation']['promoted']
    assert result['evaluation']['statistics']['truncated'] == 2
    assert base.read_bytes() == before
    candidate, identity = load_weights(Path(result['candidate']))
    assert identity.startswith('human-')
    assert any(not torch.equal(p, candidate.state_dict()[k]) for k, p in model.state_dict().items())
    payload = torch.load(result['candidate'], weights_only=True)
    assert payload['human_dataset_sha256'] == file_hash(Path(result['dataset']))
    assert payload['human_training']['positions'] == 2
    assert not (tmp_path/'data/replay_buffer/training.sqlite3').exists()
    assert not (tmp_path/'models/fly_latest.pt').exists()


def test_cancelled_human_candidate_is_partial_and_never_evaluated(tmp_path):
    archive(tmp_path, mate_session())
    model = PolicyValueNetwork(NetworkSpec(channels=8, residual_blocks=1))
    base = tmp_path/'base.pt'
    save_weights(model, base)
    result = train_human_candidate(tmp_path, base, Config(device='cpu'), cancel=lambda: True)
    assert result['status'] == 'interrupted_training' and not result['updates']
    assert result['evaluation'] is None
    assert not (tmp_path/'models/fly_best.pt').exists()
    assert torch.load(result['candidate'], weights_only=True)['human_training_status'] == 'partial'


def test_archive_worker_retains_failed_data_for_retry(tmp_path, monkeypatch):
    import fly_chess.game.human_data as storage
    actual = storage.save_human_game
    def fail(*args, **kwargs):
        raise OSError('disk unavailable')
    worker = HumanArchiveWorker()
    session = mate_session()
    try:
        monkeypatch.setattr(storage, 'save_human_game', fail)
        worker.submit(tmp_path, session, consent=True)
        try:
            worker.pending[0][0].result(timeout=5)
        except OSError:
            pass
        assert 'disk unavailable' in worker.drain()[0]
        assert len(worker.failed) == 1
        monkeypatch.setattr(storage, 'save_human_game', actual)
        worker.retry()
        worker.pending[0][0].result(timeout=5)
        assert not worker.drain() and not worker.failed
        assert (tmp_path/f'data/human_games/{session.game_id}.pgn').exists()
    finally:
        worker.close()


def test_gui_opt_in_saves_only_on_leaving_and_preserves_selected_model(tmp_path):
    pg.init()
    app = Application(tmp_path)
    try:
        app.session = mate_session()
        app.save_human = True
        game_id = app.session.game_id
        app.draw()
        assert not (tmp_path/'data/human_games').exists()
        app.new_game()
        app.archives.pending[0][0].result(timeout=5)
        assert not app.archives.drain()
        assert (tmp_path/f'data/human_games/{game_id}.pgn').exists()
        app.screen = 'train'
        app.training.pane = 'Human'
        app.draw()
        app.screen = 'settings'
        app.draw()
    finally:
        app.close()
        pg.quit()


def test_close_rejects_late_ai_and_prevents_changes_after_archive_snapshot(tmp_path, monkeypatch):
    from threading import Event
    import fly_chess.game.human_data as storage
    actual = storage.save_human_game
    release = Event()
    def delayed(*args, **kwargs):
        assert release.wait(5)
        return actual(*args, **kwargs)
    monkeypatch.setattr(storage, 'save_human_game', delayed)
    pg.init()
    app = Application(tmp_path)
    try:
        app.new_game()
        app.save_human = True
        app.session.move_human('e2e4')
        stale = app.session.token
        game_id = app.session.game_id
        app.handle_event(pg.event.Event(pg.QUIT))
        assert app.closing
        app.worker.publish({'token': stale, 'kind': 'result', 'move': chess.Move.from_uci('e7e5')})
        app.draw()
        app.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=(260, 851)), (260, 851))
        app.tick()
        assert app.session.game_id == game_id
        assert len(app.session.game.board.move_stack) == 1
        release.set()
        for future, _, _ in app.archives.pending:
            future.result(timeout=5)
        app.tick()
        assert not app.running
        with (tmp_path/f'data/human_games/{game_id}.pgn').open() as stream:
            record = chess.pgn.read_game(stream)
        assert list(record.mainline_moves()) == [chess.Move.from_uci('e2e4')]
        assert record.headers['Result'] == '*'
    finally:
        release.set()
        app.close()
        pg.quit()
