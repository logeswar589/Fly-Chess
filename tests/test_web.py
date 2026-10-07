import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import chess
import pytest

from fly_chess.config import Config
from fly_chess.core.rules import ChessGame
from fly_chess.game.session import HumanSession
from fly_chess.web.server import BrowserApplication, Server


@pytest.fixture
def browser_server(tmp_path):
    config = Config(device='cpu', network_channels=8, residual_blocks=1, mcts_simulations=2,
                    games_per_generation=2, max_game_plies=8, updates_per_generation=2,
                    batch_size=4, replay_buffer_size=16, evaluation_enabled=False)
    application = BrowserApplication(tmp_path, config)
    server = Server(('127.0.0.1', 0), application, 'test-local-access')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(path='/api/state', body=None, authorized=True, origin=None):
        headers = {'Authorization': 'Bearer test-local-access'} if authorized else {}
        if origin:
            headers['Origin'] = origin
        req = Request(f'http://127.0.0.1:{server.server_port}'+path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers=headers)
        with urlopen(req, timeout=10) as response:
            data = response.read()
            return json.loads(data) if response.headers['Content-Type'] == 'application/json' else data
    try:
        yield application, call
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        application.close()


def wait_state(call, predicate, timeout=45):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        state = call()
        assert not state['error'], state['error']
        if predicate(state):
            return state
        time.sleep(.05)
    raise AssertionError('Browser state did not reach expected condition')


def test_http_access_controls_and_static_assets(browser_server):
    app, call = browser_server
    for path in ('/api/state',):
        with pytest.raises(HTTPError) as exc:
            call(path, authorized=False)
        assert exc.value.code == 401
    with pytest.raises(HTTPError) as exc:
        call('/api/action', {'action': 'resign'}, origin='https://unrelated.example')
    assert exc.value.code == 403
    for path in ('/models/fly_latest.pt', '/../../pyproject.toml', '/logs/fly_chess.log'):
        with pytest.raises(HTTPError) as exc:
            call(path)
        assert exc.value.code == 404
    assert b'A mind' in call('/', authorized=False)
    assert b'pointermove' in call('/app.js', authorized=False)
    assert call('/Sans.ttf', authorized=False)
    with pytest.raises(HTTPError):
        call('/api/action', {'action': 'new', 'model': '../private.pt'})


def test_browser_play_real_inference_stale_moves_and_promotion(browser_server):
    app, call = browser_server
    initial = wait_state(call, lambda s: bool(s['brain']))
    assert initial['brain']['edges']
    call('/api/action', {'action': 'move', 'move': 'e2e4', 'version': initial['version']})
    with pytest.raises(HTTPError):
        call('/api/action', {'action': 'move', 'move': 'd2d4', 'version': initial['version']})
    replied = wait_state(call, lambda s: len(s['history']) >= 2 and s['human_turn'])
    assert replied['history'][0] == 'e4'
    call('/api/action', {'action': 'undo'})
    assert call()['history'] == []
    with app.lock:
        app.app.worker.stop()
        app.app.session = HumanSession(game=ChessGame.from_fen('7k/P7/8/8/8/8/8/7K w - - 0 1', claim_draws=False))
        app.app.requested = None
    state = call()
    assert 'a7a8n' in state['legal']
    call('/api/action', {'action': 'move', 'move': 'a7a8n', 'version': state['version']})
    assert chess.Board(call()['fen']).piece_at(chess.A8).piece_type == chess.KNIGHT


def test_browser_training_save_reload_and_real_brain_terms(browser_server):
    app, call = browser_server
    wait_state(call, lambda s: bool(s['brain']))
    call('/api/action', {'action': 'train_start', 'generations': 1})
    final = wait_state(call, lambda s: s['training']['progress'].get('completed_generations') == 1 and not s['training']['busy'])
    train = final['training']
    assert train['progress']['training_steps'] == 2
    assert train['before']['edges'] and train['after']['edges']
    assert train['sample_fen']
    assert train['before']['nodes'] != train['after']['nodes']
    assert train['watch']['record']['moves']
    call('/api/action', {'action': 'train_load'})
    loaded = wait_state(call, lambda s: not s['training']['busy'] and s['training']['progress'].get('status') == 'loaded')
    assert loaded['training']['progress']['training_steps'] == 2
    call('/api/action', {'action': 'watch', 'step': 1})
    assert call()['training']['watch']['ply'] == 1
