"""Token-protected HTTP adapter. One owner serializes all application state.

No model files, replay archives or arbitrary filesystem paths are exposed by HTTP.
The browser controls one shared match and training workspace, not separate accounts.
"""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

import chess

from fly_chess.ui.app import Application
from fly_chess.ui.brain_view import BrainView


def graph(snapshot):
    if not snapshot:
        return None
    nodes = [{'id': f'{name}|{index}', 'xyz': xyz, 'layer': name, 'index': index, 'value': value}
             for xyz, name, index, value in BrainView.nodes(snapshot)]
    return {'nodes': nodes, 'edges': snapshot.get('connections', []),
            'timestamp': snapshot.get('timestamp_utc'), 'fen': snapshot.get('fen'),
            'model': snapshot.get('model_id')}


class BrowserApplication:
    def __init__(self, workspace, config, weights=None):
        self.app = Application(workspace, weights, config)
        self.app.training.interval = 1
        self.app.new_game()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = ''
        self._graphs = deque(maxlen=6)
        self.thread = threading.Thread(target=self._tick, name='fly-browser-owner', daemon=True)
        self.thread.start()

    def _tick(self):
        while not self.stop.wait(.04):
            with self.lock:
                try:
                    self.app.tick()
                except Exception as exc:
                    self.error = str(exc)

    def _graph(self, capture):
        if not capture:
            return None
        for original, value in self._graphs:
            if original is capture:
                return value
        result = graph(capture)
        self._graphs.append((capture, result))
        return result

    def state(self):
        with self.lock:
            app, train = self.app, self.app.training
            session, board = app.session, app.session.game.board
            data = app.visible_data()
            ev = data['evaluation'] if data else None
            sample = train.frozen or train.state.get('inspection') or {}
            metrics = train.state.get('metrics', [])[-100:]
            return {
                'fen': board.fen(), 'version': list(session.token), 'human': 'white' if session.human_color else 'black',
                'human_turn': session.human_turn, 'finished': session.finished,
                'status': session.manual_result or (board.result() if session.finished else 'Your move' if session.human_turn else 'Fly is thinking'),
                'legal': [m.uci() for m in board.legal_moves] if session.human_turn else [],
                'history': session.history(), 'can_undo': session.can_undo,
                'can_claim': session.human_turn and board.can_claim_draw(), 'last_move': board.peek().uci() if board.move_stack else None,
                'models': [p.relative_to(app.workspace).as_posix() for p in sorted((app.workspace/'models').glob('*.pt'))],
                'model': app.match_model_id, 'difficulty': app.difficulty, 'adapt': app.adapt,
                'save_human': app.save_human, 'telemetry': app.telemetry, 'frozen': app.frozen,
                'brain': self._graph(ev.inspection) if ev else None,
                'brain_source': {'fen': data['fen'], 'ply': data['ply'], 'current': data['token'] == session.token} if data else None,
                'value': ev.value if ev else None, 'search': data.get('search') if data else None,
                'opponent': data.get('profile') if data else None,
                'error': self.error or app.error or train.error,
                'training': {'busy': train.worker.busy, 'notice': train.notice, 'progress': train.state.get('progress', {}),
                    'before': self._graph(sample.get('before')), 'after': self._graph(sample.get('after')),
                    'sample_step': sample.get('step'), 'sample_fen': sample.get('fen'),
                    'interval': train.interval, 'diagnostics': train.telemetry, 'activations': train.activations,
                    'gradients': train.gradients, 'frozen': bool(train.frozen),
                    'metrics': [{k: m.get(k) for k in ('step', 'loss', 'policy_loss', 'value_loss')} for m in metrics],
                    'learning': sample.get('learning'), 'human_report': train.human_report,
                    'evaluations': train.state.get('evaluations', [])[-5:],
                    'watch': {'record': train.watch, 'fen': train.watch_game.board.fen(), 'ply': train.watch_ply,
                              'playing': train.watch_playing, 'speed': train.speed_index,
                              'brain': self._graph(train.watch_data['evaluation'].inspection) if train.watch_data else None}},
            }

    def action(self, message):
        if not isinstance(message, dict):
            raise ValueError('Expected a JSON object')
        with self.lock:
            app, train = self.app, self.app.training
            action = message.get('action')
            if action == 'move':
                if message.get('version') != list(app.session.token):
                    raise ValueError('The position changed. Select your move again.')
                move = chess.Move.from_uci(str(message.get('move', '')))
                if not app.session.human_turn or move not in app.session.game.board.legal_moves:
                    raise ValueError('That move is not legal in the current position.')
                app.play_move(move)
            elif action == 'new':
                difficulty = message.get('difficulty', 1)
                if type(difficulty) is not int or difficulty not in range(4):
                    raise ValueError('Choose a supported search budget')
                model = message.get('model', '')
                choices = {p.relative_to(app.workspace).as_posix(): p for p in (app.workspace/'models').glob('*.pt')}
                if model and model not in choices:
                    raise ValueError('Choose a checkpoint listed in this workspace')
                if message.get('side', 'white') not in ('white', 'black'):
                    raise ValueError('Choose White or Black')
                app.weights = choices.get(model)
                app.human_color = message.get('side', 'white') == 'white'
                app.difficulty = difficulty
                app.new_game()
            elif action in ('undo', 'resign', 'claim_draw', 'toggle_freeze'):
                getattr(app, action)()
            elif action == 'settings':
                for field in ('adapt', 'telemetry', 'save_human'):
                    if field in message:
                        if type(message[field]) is not bool:
                            raise ValueError('Settings must be true or false')
                        if field == 'adapt' and message[field] != app.adapt:
                            app.toggle_adapt()
                        elif field == 'telemetry' and message[field] != app.telemetry:
                            app.toggle_telemetry()
                        else:
                            setattr(app, field, message[field])
            elif action in ('train_start', 'train_load'):
                count = message.get('generations', 1)
                if type(count) is not int or count not in (1, 2, 5, 10):
                    raise ValueError('Choose 1, 2, 5 or 10 generations')
                if train.worker.busy:
                    raise ValueError('Training is already running')
                train.generations = count
                train.start(load_only=action == 'train_load')
            elif action in ('train_pause', 'train_stop', 'train_save'):
                train.request(action.split('_')[1])
            elif action == 'train_settings':
                interval = message.get('interval', train.interval)
                if type(interval) is not int or interval not in (1, 8, 32):
                    raise ValueError('Invalid diagnostic interval')
                for field in ('diagnostics', 'activations', 'gradients'):
                    if field in message and type(message[field]) is not bool:
                        raise ValueError('Settings must be true or false')
                train.interval = interval
                train.telemetry = message.get('diagnostics', train.telemetry)
                train.activations = message.get('activations', train.activations)
                train.gradients = message.get('gradients', train.gradients)
                train.configure()
            elif action == 'train_freeze':
                train.toggle_freeze()
            elif action == 'watch':
                speed = message.get('speed', train.speed_index)
                step = message.get('step', 0)
                if type(speed) is not int or speed not in range(5) or type(step) is not int or abs(step) > 512:
                    raise ValueError('Invalid replay control')
                train.pane = 'Watch'
                train.speed_index = speed
                if step:
                    train.manual_step(step)
                if 'playing' in message:
                    if type(message['playing']) is not bool:
                        raise ValueError('Invalid playback state')
                    train.watch_playing = message['playing']
            elif action == 'train_human':
                if train.worker.busy:
                    raise ValueError('Stop current training first')
                base = train.human_base(app)
                if not base:
                    raise ValueError('A base checkpoint is required')
                train.start_human(base)
            elif action == 'retry_save':
                app.retry_archives()
            else:
                raise ValueError('Unknown action')
            return {'ok': True}

    def close(self):
        self.stop.set()
        self.thread.join()
        self.app.worker.stop()
        self.app.session.invalidate()
        self.app.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, application, token):
        self.application, self.token = application, token
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Never log access tokens or game payloads.

    def send(self, status, data, content_type='application/json'):
        if not isinstance(data, bytes):
            data = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        expected = 'Bearer '+self.server.token
        return secrets.compare_digest(self.headers.get('Authorization', ''), expected)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/state':
            if not self.authorized():
                return self.send(401, {'error': 'Open the access link printed by the server.'})
            return self.send(200, self.server.application.state())
        root = Path(__file__).parent/'static'
        assets = {'/': root/'index.html', '/app.js': root/'app.js', '/style.css': root/'style.css',
                  '/Sans.ttf': Path(__file__).parents[1]/'ui/assets/Sans.ttf',
                  '/Mono.ttf': Path(__file__).parents[1]/'ui/assets/Mono.ttf'}
        if path not in assets:
            return self.send(404, {'error': 'Not found'})
        file = assets[path]
        return self.send(200, file.read_bytes(), mimetypes.guess_type(str(file))[0] or 'application/octet-stream')

    def do_POST(self):
        if urlsplit(self.path).path != '/api/action':
            return self.send(404, {'error': 'Not found'})
        if not self.authorized():
            return self.send(401, {'error': 'Access token missing or expired.'})
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.send(403, {'error': 'Cross-origin actions are not allowed'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                return self.send(413, {'error': 'Invalid request size'})
            message = json.loads(self.rfile.read(size))
            result = self.server.application.action(message)
            self.send(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            self.send(400, {'error': str(exc)})
        except Exception:
            self.send(500, {'error': 'Action failed. Check the server terminal or application status.'})


def run(workspace, config, *, host='127.0.0.1', port=8765, weights=None):
    token = secrets.token_urlsafe(32)
    application = BrowserApplication(Path(workspace).resolve(), config, weights)
    try:
        with Server((host, port), application, token) as server:
            port = server.server_port
            print(f'Fly / Chess - browser edition\nOpen http://127.0.0.1:{port}/#token={token}', flush=True)
            if host == '0.0.0.0':
                addresses = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
                for address in addresses:
                    print(f'LAN: http://{address}:{port}/#token={token}', flush=True)
                print('Access links control this shared workspace. Use on a trusted LAN or private VPN.', flush=True)
            print('Keep this terminal open. Ctrl+C saves/stops workers at a safe boundary.', flush=True)
            try:
                server.serve_forever(poll_interval=.2)
            except KeyboardInterrupt:
                pass
    finally:
        print('Saving and closing Fly workers...', flush=True)
        application.close()
