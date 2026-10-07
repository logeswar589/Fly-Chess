"""Consent-gated PGNs and a separate, strictly validated human dataset."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile

import chess
import chess.pgn
import numpy as np

from fly_chess.core.actions import encode_move
from fly_chess.core.encoding import encode_board
from fly_chess.core.rules import ChessGame
from fly_chess.selfplay.records import Sample


def save_human_game(root, game, *, game_id, human_color, manual_result=None,
                    model_id='unknown', checkpoint='unavailable', adaptation=False, consent=False):
    if not consent or not game.board.move_stack:
        return None
    if re.fullmatch(r'[a-f0-9]{32}', game_id) is None:
        raise ValueError('Invalid human game identity')
    board = game.board
    result = board.result(claim_draw=False)
    termination = game.result.termination if game.result else 'unfinished'
    if manual_result == 'Draw claimed.':
        if not board.can_claim_draw():
            raise ValueError('Claimed human draw is not supported by the position')
        result, termination = '1/2-1/2', 'claimed_draw'
    elif manual_result == 'You resigned. Fly wins.':
        result, termination = ('0-1' if human_color else '1-0'), 'resignation'
    elif manual_result is not None:
        raise ValueError('Unknown human result')
    record = chess.pgn.Game.from_board(board)
    clean = lambda value: ' '.join(str(value).split())[:256]
    record.headers.update({'Event': 'Fly / Human match', 'Site': 'Local',
        'Date': datetime.now(timezone.utc).strftime('%Y.%m.%d'), 'Round': '-',
        'White': 'Human' if human_color else 'Fly', 'Black': 'Fly' if human_color else 'Human',
        'Result': result, 'Termination': termination, 'FlySchema': '1', 'FlySaveConsent': 'true',
        'FlySource': 'human-vs-fly', 'FlyGameId': game_id,
        'FlyHumanColor': 'white' if human_color else 'black', 'FlyModelId': clean(model_id),
        'FlyCheckpoint': clean(checkpoint), 'FlyAdaptation': str(adaptation).lower(),
        'FlyOutcome': 'incomplete' if result == '*' else 'completed'})
    text = str(record)+'\n\n'
    target = Path(root) / 'data/human_games' / f'{game_id}.pgn'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, mode='w', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return target


@dataclass
class HumanDataset:
    samples: list
    keys: list
    sources: list
    excluded_incomplete: int


def load_human_dataset(root, *, max_positions=4096):
    if type(max_positions) is not int or not 1 <= max_positions <= 100000:
        raise ValueError('Human dataset position limit must be in [1, 100000]')
    samples, keys, sources, incomplete = [], [], [], 0
    for path in sorted((Path(root)/'data/human_games').glob('*.pgn')):
        if path.stat().st_size > 2_000_000:
            raise ValueError(f'Human PGN exceeds the per-game size limit: {path.name}')
        raw = path.read_bytes()
        stream = io.StringIO(raw.decode('utf-8'))
        record = chess.pgn.read_game(stream)
        if record is None or record.errors or chess.pgn.read_game(stream) is not None:
            raise ValueError(f'Invalid or multiple games in {path.name}')
        headers = record.headers
        required = {'FlySchema': '1', 'FlySaveConsent': 'true', 'FlySource': 'human-vs-fly'}
        if any(headers.get(k) != v for k, v in required.items()):
            raise ValueError(f'Missing consent/provenance in {path.name}')
        if headers.get('FlyGameId') != path.stem or headers.get('FlyHumanColor') not in ('white', 'black'):
            raise ValueError(f'Invalid human-game identity/color in {path.name}')
        if headers.get('Result') == '*':
            incomplete += 1
            continue
        if headers.get('Result') not in ('1-0', '0-1', '1/2-1/2') or headers.get('FlyOutcome') != 'completed':
            raise ValueError(f'Invalid completed outcome in {path.name}')
        human_color = headers['FlyHumanColor'] == 'white'
        game = ChessGame(record.board(), claim_draws=False)
        positions = []
        available = 0
        remaining = max_positions-len(samples)
        result = {'1-0': 1., '0-1': -1., '1/2-1/2': 0.}[headers['Result']]
        for ply, move in enumerate(record.mainline_moves()):
            if game.board.turn == human_color:
                available += 1
                if len(positions) < remaining:
                    actions = game.actions()
                    action = encode_move(move)
                    policy = (actions == action).astype(np.float32)
                    positions.append((ply, Sample(encode_board(game.board), action, actions, policy,
                        human_color, 0., 'human_demonstration', result if human_color else -result)))
            game.push(move)
        termination = headers.get('Termination')
        if termination == 'resignation':
            expected = '0-1' if human_color else '1-0'
            valid = headers['Result'] == expected and game.result is None
        elif termination == 'claimed_draw':
            valid = headers['Result'] == '1/2-1/2' and game.board.can_claim_draw()
        else:
            valid = game.result is not None and headers['Result'] == game.board.result() and termination == game.result.termination
        if not valid:
            raise ValueError(f'Outcome is inconsistent with legal history in {path.name}')
        selected = positions
        if selected:
            samples.extend(sample for _, sample in selected)
            keys.extend([headers['FlyGameId'], ply] for ply, _ in selected)
            sources.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                'pgn': raw.decode('utf-8'), 'positions_used': len(selected),
                'positions_available': available, 'result': headers['Result'],
                'termination': termination, 'opponent_model': headers.get('FlyModelId', 'unknown')})
        if len(samples) >= max_positions:
            break
    if not samples:
        raise ValueError('No completed, consented human decisions are available. Incomplete games are excluded.')
    return HumanDataset(samples, keys, sources, incomplete)


class HumanArchiveWorker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-pgn')
        self.pending = []
        self.failed = []

    def submit(self, root, session, **metadata):
        if self.failed:
            raise RuntimeError('A human-game save failed. Retry the pending archive before leaving this match.')
        if len(self.pending) >= 4:
            raise RuntimeError('Human-game archive queue is full; wait for pending saves before leaving this match.')
        args = (root, session.game.copy())
        kwargs = dict(game_id=session.game_id, human_color=session.human_color, manual_result=session.manual_result, **metadata)
        self.pending.append((self.executor.submit(save_human_game, *args, **kwargs), args, kwargs))

    def drain(self):
        errors = []
        remaining = []
        for future, args, kwargs in self.pending:
            if not future.done():
                remaining.append((future, args, kwargs))
            else:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f'Human game was not saved: {exc}')
                    self.failed.append((args, kwargs))
        self.pending = remaining
        return errors

    def retry(self):
        for args, kwargs in self.failed:
            self.pending.append((self.executor.submit(save_human_game, *args, **kwargs), args, kwargs))
        self.failed = []

    def close(self):
        self.executor.shutdown(wait=True)
        return self.drain()
