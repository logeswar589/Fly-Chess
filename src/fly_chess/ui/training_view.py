from dataclasses import replace
from pathlib import Path
import time
import chess
import pygame as pg

from fly_chess.config import Config, load_config
from fly_chess.core.rules import ChessGame
from fly_chess.ui.board import draw_board
from fly_chess.ui.theme import BG, INK, MUTED, RULE
from fly_chess.ui.training_worker import TrainingWorker
from fly_chess.ui.worker import BrainWorker


SPEEDS = (.25, .5, 1., 2., None)


def series_for(state):
    """Persisted observations only. None remains a gap, never a zero rating."""
    metrics, evaluations = state.get('metrics', []), state.get('evaluations', [])
    return {
        'loss': [[(m['step'], m.get(key)) for m in metrics] for key in ('loss', 'policy_loss', 'value_loss')],
        'win_rate': [[(e['generation'], (e.get('statistics') or {}).get('candidate_win_rate')) for e in evaluations]],
        'length': [[(m.get('games_played', m['step']), m.get('average_game_length')) for m in metrics]],
        # Comparisons against changing incumbents cannot form an absolute Elo series.
        'rating': [[(e.get('games_played', 0), None) for e in evaluations]],
    }


def plot(p, title, series, x, y, w=640, h=82, empty='No observations yet'):
    p.label(title, x, y)
    values = [(a, b) for line in series for a, b in line if b is not None]
    p.line(x, y+h-17, w)
    if not values:
        p.text(empty, x, y+29, 16, MUTED)
        return
    x0, x1 = min(a for a, b in values), max(a for a, b in values)
    low, high = min(b for a, b in values), max(b for a, b in values)
    span = max(high-low, abs(high)*.1, .001)
    def point(a, b):
        return (x+70+(a-x0)/max(1, x1-x0)*(w-88), y+h-22-(b-low)/span*(h-44))
    for index, line in enumerate(series):
        previous = None
        for a, b in line:
            if b is None:
                previous = None
                continue
            position = point(a, b)
            color = (INK, MUTED, (95, 95, 93))[index % 3]
            if previous:
                p.segment(color, previous, position, 2 if index == 0 else 1)
            p.circle(color, position, 2)
            previous = position
    p.text(f'{high:.3g}', x, y+22, 12, MUTED, True)
    p.text(f'{low:.3g}', x, y+h-34, 12, MUTED, True)
    p.text(str(x0), x+70, y+h-12, 11, MUTED, True)
    p.text(str(x1), x+w-48, y+h-12, 11, MUTED, True)


def heatmap(p, tensor, x, y, label, channel=0, scale=None):
    values = tensor.get('sample_values', [])[channel*64:(channel+1)*64] if tensor else []
    p.label(label, x, y)
    if not values:
        p.text('Unavailable', x, y+43, 15, MUTED)
        return
    limit = scale or max(max(abs(v) for v in values), 1e-8)
    for i, value in enumerate(values):
        gray = round(127+120*value/limit)
        row, col = divmod(i, 8)
        p.rect((gray, gray, gray), (x+col*18, y+27+(7-row)*18, 16, 16))
    p.text(f'±{limit:.4g}', x, y+179, 12, MUTED, True)


class TrainingView:
    def __init__(self, root, config=None):
        self.root = Path(root)
        self.config = config or Config()
        self.worker = TrainingWorker()
        self.brain = BrainWorker()
        self.state = {}
        self.error = ''
        self.notice = 'Ready. Start a new run, or load a full checkpoint.'
        self.resume = self.root / 'models/fly_latest.pt'
        self.generations = 1
        self.pane = 'Brain'
        from fly_chess.ui.brain_view import BrainView
        self.brain_view = BrainView()
        self.telemetry = True
        self.activations = True
        self.gradients = True
        self.tensor_index = 0
        self.interval = 8
        self.module = 'stem'
        self.channel = 0
        self.frozen = None
        self.watch = None
        self.pending_watch = None
        self.watch_game = ChessGame()
        self.watch_ply = 0
        self.watch_playing = True
        self.speed_index = 2
        self.last_step = time.monotonic()
        self.watch_revision = 0
        self.watch_requested = None
        self.watch_data = None
        self.loading = False
        self.closing = False
        self.mode = 'selfplay'
        self.human_report = None

    def start(self, load_only=False):
        if self.worker.busy or self.closing:
            return
        resume = self.resume if self.resume.exists() else None
        if load_only and resume is None:
            self.error = 'No full checkpoint found. Start training first, or drop a checkpoint here.'
            return
        if self.worker.start(self.root, self.config, resume=resume, generations=self.generations,
                telemetry=self.telemetry, interval=self.interval, module=self.module if self.activations else None,
                load_only=load_only, gradients=self.gradients):
            self.loading = load_only
            self.mode = 'selfplay'
            self.error = ''
            self.notice = 'Loading saved run…' if load_only else 'Starting worker…'

    def request(self, action):
        self.worker.request(action)
        self.notice = {'pause': 'Pause requested / finishes the current wave or minibatch.',
                       'stop': 'Stop requested / saving at the next safe boundary.',
                       'save': 'Save requested / waiting for a durable boundary.'}[action]
        if self.mode == 'human':
            self.notice = 'Stopping human training / saving the candidate without promotion.'

    def start_human(self, base):
        if self.worker.busy or self.closing:
            return
        if self.worker.start(self.root, self.config, human_base=base):
            self.mode = 'human'
            self.loading = False
            self.human_report = None
            self.error = ''
            self.notice = 'Starting explicit human-data training…'

    def human_base(self, app):
        candidates = [app.weights, self.root/'models/fly_best.pt', self.root/'models/fly_latest.pt']
        return next((p for p in candidates if p is not None and Path(p).is_file()), None)

    def drop(self, path):
        if self.worker.busy:
            self.error = 'Pause or stop training before changing its checkpoint or configuration.'
            return
        try:
            if path.suffix.lower() == '.toml':
                self.config = load_config(path)
                self.notice = 'Configuration loaded. Existing runs keep their saved configuration.'
            elif path.suffix.lower() == '.pt':
                self.resume = path
                self.start(load_only=True)
            else:
                self.error = 'Use a full .pt checkpoint or a .toml configuration.'
        except ValueError as exc:
            self.error = str(exc)

    def configure(self):
        self.worker.configure(self.telemetry, self.interval, self.module if self.activations else None, self.gradients)

    def toggle_telemetry(self):
        self.telemetry = not self.telemetry
        self.configure()

    def toggle_activations(self):
        self.activations = not self.activations
        self.configure()

    def toggle_gradients(self):
        self.gradients = not self.gradients
        self.configure()

    def change_interval(self):
        choices = (1, 8, 32)
        self.interval = choices[(choices.index(self.interval)+1) % len(choices)]
        self.configure()

    def change_module(self, delta):
        names = [m['name'] for m in self.state.get('inventory', [])]
        if names:
            self.module = names[((names.index(self.module) if self.module in names else 0)+delta) % len(names)]
            self.channel = 0
            self.configure()
            self.brain.stop()
            self.watch_revision += 1
            self.watch_requested = None

    def toggle_freeze(self):
        self.frozen = None if self.frozen else self.state.get('inspection')

    def set_watch(self, record):
        self.watch = record
        self.watch_game = ChessGame.from_fen(record['root_fen'])
        for move in record['prefix_moves']:
            self.watch_game.push(move)
        self.watch_ply = 0
        self.watch_revision += 1
        self.watch_requested = None
        self.watch_data = None
        self.brain.stop()
        self.last_step = time.monotonic()

    def step_watch(self, amount=1):
        if not self.watch:
            return
        target = max(0, min(len(self.watch['moves']), self.watch_ply+amount))
        while self.watch_ply < target:
            self.watch_game.push(self.watch['moves'][self.watch_ply])
            self.watch_ply += 1
        while self.watch_ply > target:
            self.watch_game.undo()
            self.watch_ply -= 1
        self.watch_revision += 1
        self.watch_requested = None
        self.brain.stop()

    def tick(self):
        for message in self.worker.drain():
            if message['kind'] == 'error':
                self.error = message['error']
                self.notice = 'Worker failed. Reload the last durable checkpoint before retrying.'
                continue
            if message['kind'] == 'human':
                self.human_report = message['report']
                self.notice = 'Human data / '+message['report']['status']
                continue
            self.state = message
            self.config = Config(**message['configuration'])
            self.resume = Path(message['latest'])
            self.notice = 'Saved / '+message['progress']['status']
            record = message.get('watch')
            if record and (not self.watch or record['game_id'] != self.watch['game_id']):
                if not self.watch or self.watch_ply == len(self.watch['moves']):
                    self.set_watch(record)
                else:
                    self.pending_watch = record
        now = time.monotonic()
        if self.pane == 'Watch' and self.watch and self.watch_playing and not self.closing:
            speed = SPEEDS[self.speed_index]
            if speed is None:
                if self.watch_ply < len(self.watch['moves']):
                    self.step_watch(len(self.watch['moves']))
            elif now-self.last_step >= 1/speed:
                self.step_watch()
                self.last_step = now
            if self.watch_ply == len(self.watch['moves']) and self.pending_watch:
                self.set_watch(self.pending_watch)
                self.pending_watch = None
        token = (self.watch['game_id'], self.watch_revision) if self.watch else None
        for message in self.brain.drain():
            if message['token'] == token:
                if message['kind'] == 'error':
                    self.error = message['error']
                elif message['evaluation'].model_id == self.watch['model_id']:
                    self.watch_data = message
        if (not self.closing and self.pane == 'Watch' and self.watch and self.telemetry and self.activations
                and SPEEDS[self.speed_index] is not None and not self.watch_game.result
                and self.watch_requested != token and not self.brain.busy):
            run_id, step = self.watch['model_id'].split(':step-')
            weights = self.root / f'models/selfplay/{run_id}-{step}.pt'
            if weights.exists() and self.brain.submit(token=token, game=self.watch_game,
                    human_color=True, weights=weights, module=self.module):
                self.watch_requested = token

    def draw(self, app):
        p, button = app.p, app.button
        state, progress = self.state, self.state.get('progress', {})
        inspection = self.frozen or state.get('inspection')
        p.label('02 / TRAINING LAB', 56, 112)
        p.text('Watch Fly' if self.pane == 'Watch' else 'Your game record' if self.pane == 'Human' else 'A position. An update.', 56, 145, 30)
        board = app.session.game.board if self.pane == 'Human' else self.watch_game.board if self.pane == 'Watch' else chess.Board(inspection['fen']) if inspection and inspection.get('fen') else chess.Board()
        draw_board(p, board)
        if inspection and not inspection.get('fen') and self.pane not in ('Watch', 'Human'):
            app.p.rect(BG, (56, 194, 608, 608))
            p.text('Source board unavailable', 90, 440, 28)
            p.text('The optional game archive is missing or invalid.', 90, 494, 17, MUTED)
        if self.pane == 'Watch':
            button('Pause' if self.watch_playing else 'Play', 56, 839, 88, lambda: setattr(self, 'watch_playing', not self.watch_playing))
            for i, name in enumerate(('¼×', '½×', '1×', '2×', 'MAX')):
                button(name, 154+i*72, 839, 62, lambda n=i: setattr(self, 'speed_index', n), active=self.speed_index == i)
            button('<', 524, 839, 60, lambda: self.manual_step(-1), enabled=bool(self.watch))
            button('>', 594, 839, 70, lambda: self.manual_step(1), enabled=bool(self.watch))
        elif self.pane == 'Human':
            p.text('Current human match / '+('saving enabled' if app.save_human else 'saving disabled'), 56, 841, 19, MUTED)
        else:
            p.text(('Frozen replay sample' if self.frozen else 'Sampled replay position') if inspection else 'Awaiting a sampled optimizer step.', 56, 836, 19, MUTED)
            if inspection:
                side = ('White' if board.turn else 'Black')+' to move' if inspection.get('fen') else 'Source board unavailable'
                p.text(side+' / '+inspection.get('after', {}).get('timestamp_utc', '')[11:19]+' UTC', 56, 865, 12, MUTED, True)
                p.text(f'Step {inspection["step"]} / '+str(inspection['sample_key'])[:64], 56, 888, 12, MUTED, True)
        busy = self.worker.busy
        button('Resume' if self.resume.exists() else 'Start', 744, 111, 132, self.start, enabled=not busy and not self.closing)
        button('Pause', 886, 111, 94, lambda: self.request('pause'), enabled=busy and not self.loading and self.mode == 'selfplay')
        button('Stop', 990, 111, 90, lambda: self.request('stop'), enabled=busy and not self.loading)
        button('Save', 1090, 111, 90, lambda: self.request('save'), enabled=busy and not self.loading and self.mode == 'selfplay')
        button('Load', 1190, 111, 90, lambda: self.start(load_only=True), enabled=not busy)
        button(f'{self.generations} gen', 1290, 111, 94, self.cycle_generations, enabled=not busy)
        p.text(self.notice[:80], 744, 169, 15, MUTED)
        stage = progress.get('stage', 'ready')
        if self.mode == 'human':
            stage = (self.human_report or {}).get('status', 'starting')
        stage_label = stage.upper() if busy else 'NEXT / '+stage.upper()
        if self.mode == 'human':
            stage_label = 'HUMAN / '+{'training': 'LEARNING', 'evaluating': 'EVALUATION',
                'interrupted_training': 'STOPPED', 'interrupted_evaluation': 'EVAL PAUSED',
                'complete': 'COMPLETE'}.get(stage, 'STARTING')
        p.text(stage_label, 744, 204, 29)
        p.text('Checkpointed boundaries', 1110, 218, 15, MUTED)
        p.line(744, 250, 640)
        fields = [('GENERATION', progress.get('generation', '—')), ('GAMES', progress.get('games_played', '—')),
                  ('UPDATES', progress.get('training_steps', '—')), ('ACTIVE TIME', f'{progress.get("active_seconds", 0)/60:.1f}m')]
        if self.mode == 'human':
            human = self.human_report or {}
            fields = [('EPOCHS', human.get('epochs', 1)), ('POSITIONS', human.get('positions', '—')),
                      ('UPDATES', len(human.get('updates', []))), ('DATASET', 'Human')]
        for i, (label, value) in enumerate(fields):
            p.label(label, 744+i*162, 272)
            p.text(value, 744+i*162, 298, 25, INK, True)
        for i, name in enumerate(('Brain', 'Learning', 'Stats', 'Watch', 'Config', 'Human')):
            button(name, 744+i*108, 351, 100, lambda n=name: setattr(self, 'pane', n), active=self.pane == name)
        if self.pane == 'Brain':
            view = self.brain_view
            button('Live' if self.frozen else 'Freeze', 744, 405, 94, self.toggle_freeze, enabled=bool(inspection))
            button('Before', 850, 405, 100, lambda: setattr(view, 'phase', 'before'), active=view.phase == 'before')
            button('After', 960, 405, 100, lambda: setattr(view, 'phase', 'after'), active=view.phase == 'after')
            button('Synapses', 1100, 405, 132, lambda: setattr(view, 'synapses', not view.synapses), active=view.synapses)
            button('Reset view', 1244, 405, 140, view.reset)
            p.text('Scroll: zoom / drag: rotate / hover: activation', 744, 453, 13, MUTED)
            sample = inspection or {}
            view.draw(p, sample.get(view.phase), app.mouse, y=478, height=310,
                      reference=sample.get('before' if view.phase == 'after' else 'after'))
            p.text(f'Latest sampled update / step {sample.get("step", "—")} / {view.phase}', 744, 866, 13, MUTED, True)
            p.text(f'Refreshes every {self.interval} optimizer steps; self-play precedes updates.', 744, 890, 12, MUTED)
        elif self.pane == 'Learning':
            self.draw_learning(app, inspection)
        elif self.pane == 'Stats':
            self.draw_stats(app)
        elif self.pane == 'Watch':
            self.draw_watch(app)
        elif self.pane == 'Config':
            self.draw_config(app)
        else:
            self.draw_human(app)
        if self.error:
            app.p.rect(INK, (48, 865, 1344, 44))
            p.text('ERROR / '+self.error[:142], 60, 876, 13, BG, True)

    def draw_learning(self, app, sample):
        p, button = app.p, app.button
        button('Live' if self.frozen else 'Freeze', 744, 412, 94, self.toggle_freeze, enabled=bool(sample))
        button('< module', 848, 412, 122, lambda: self.change_module(-1), enabled=not self.frozen)
        button('module >', 980, 412, 122, lambda: self.change_module(1), enabled=not self.frozen)
        button('Tensor', 1112, 412, 108, lambda: setattr(self, 'tensor_index', self.tensor_index+1))
        button('<', 1230, 412, 40, lambda: setattr(self, 'channel', max(0, self.channel-1)))
        button('>', 1344, 412, 40, lambda: setattr(self, 'channel', self.channel+1))
        p.text(str(self.channel), 1286, 423, 14, INK, True)
        if not sample:
            p.text('Waiting for a sampled update.', 744, 496, 26)
            p.text(f'Inspection interval: every {self.interval} optimizer steps.', 744, 546, 17, MUTED)
            p.text('Enable diagnostics in Config to capture learning.', 744, 584, 17, MUTED)
            p.text('The board is a placeholder until a real sample arrives.', 744, 627, 16, MUTED)
            return
        learning = sample['learning']
        p.text(sample.get('module', 'Activations disabled')[:52], 744, 468, 19)
        p.text(f'Step {sample["step"]} / same input before & after', 744, 498, 13, MUTED, True)
        before = [dict(v, name=k) for k, v in sample.get('before', {}).get('tensors', {}).items() if v['sample_values']]
        after = [dict(v, name=k) for k, v in sample.get('after', {}).get('tensors', {}).items() if v['sample_values']]
        before += [dict(v, name='weight/'+v['name']) for v in sample.get('before', {}).get('weights', [])]
        after += [dict(v, name='weight/'+v['name']) for v in sample.get('after', {}).get('weights', [])]
        index = self.tensor_index % max(1, len(before))
        a, b = before[index] if before else None, after[index] if after else None
        values = sum([t['sample_values'][self.channel*64:(self.channel+1)*64] for t in (a, b) if t], [])
        scale = max(max((abs(v) for v in values), default=0), 1e-8)
        heatmap(p, a, 744, 530, 'BEFORE', self.channel, scale)
        heatmap(p, b, 921, 530, 'AFTER', self.channel, scale)
        layer = learning['layers'].get(sample.get('module'), {})
        for i, (label, key) in enumerate((('Gradient / clipped', 'gradient_norm_after_clip'), ('Weight norm / before', 'weight_norm_before'), ('Update norm', 'update_norm'))):
            p.label(label, 1110, 540+i*64)
            value = layer.get(key)
            p.text(f'{value:.5g}' if value is not None else 'Not captured / none', 1110, 566+i*64, 15, MUTED, True)
        if b and b.get('name', '').startswith('weight/'):
            app.p.rect(BG, (1100, 530, 284, 205))
            p.label('WEIGHT SLICE / AFTER UPDATE', 1110, 540)
            selected = b['sample_values'][self.channel*64:(self.channel+1)*64]
            bins = [0]*16
            for value in selected:
                bins[min(15, max(0, int((value/scale+1)*8)))] += 1
            for i, count in enumerate(bins):
                height = round(92*count/max(1, max(bins)))
                app.p.rect(MUTED, (1110+i*16, 668-height, 13, height))
            p.text('Histogram / shared ±scale', 1110, 682, 13, MUTED)
            p.text(f'{len(selected)} shown / {b["total_elements"]} total', 1110, 711, 13, MUTED, True)
        p.text(f'Value: {learning["value_before"]:+.3f} → {learning["value_after"]:+.3f} / target {learning["target_value"]:+.0f}', 744, 744, 17, INK, True)
        from fly_chess.core.actions import decode_action
        best = sorted(range(len(learning['legal_actions'])), key=lambda i: -learning['target_policy'][i])[:3]
        p.label('TOP 3 TARGETS / PREDICTED BEFORE → AFTER', 744, 785)
        for j, i in enumerate(best):
            p.text(f'{decode_action(learning["legal_actions"][i]).uci()}  {learning["target_policy"][i]:5.1%}   {learning["policy_before"][i]:5.1%} → {learning["policy_after"][i]:5.1%}', 744, 809+j*24, 14, INK, True)
        p.text((f'{a["name"]} {a["shape"]} / slice {self.channel*64}:{self.channel*64+64}' if a else 'No activation capture')[:76], 744, 890, 12, MUTED, True)

    def draw_stats(self, app):
        p = app.p
        lines = series_for(self.state)
        plot(p, 'LOSSES / TOTAL · POLICY · VALUE / OPTIMIZER STEP', lines['loss'], 744, 414, h=96)
        plot(p, 'CANDIDATE WIN RATE / GENERATION', lines['win_rate'], 744, 530, h=88, empty='No completed evaluation outcomes')
        plot(p, 'MEAN GAME LENGTH (CUMULATIVE) / GAMES', lines['length'], 744, 636, h=88)
        plot(p, 'ABSOLUTE RATING / GAMES', lines['rating'], 744, 742, h=78, empty='Unrated. No calibrated absolute Elo exists.')
        progress = self.state.get('progress', {})
        p.text(f'White wins {progress.get("white_wins", 0)} / Black wins {progress.get("black_wins", 0)} / Draws {progress.get("draws", 0)}', 744, 840, 14, INK, True)
        p.text(f'Truncated {progress.get("truncated_games", 0)} / LR {self.state.get("learning_rate", self.config.learning_rate):.6g}', 744, 867, 14, MUTED, True)
        p.text(f'Last 1,000 updates shown / omitted {self.state.get("metrics_omitted", 0)}', 744, 893, 12, MUTED)

    def draw_watch(self, app):
        p = app.p
        if not self.watch:
            p.text('No readable archived game.', 744, 441, 26)
            p.text('Watch plays completed or truncated self-play archives.', 744, 496, 17, MUTED)
            p.text('Training continues independently of playback speed.', 744, 534, 17, MUTED)
            return
        p.label('RECORDED SELF-PLAY / NOT A LIVE SEARCH', 744, 417)
        p.text(f'Ply {self.watch_ply} / {len(self.watch["moves"])}', 744, 448, 27)
        p.text(f'{self.watch["status"]} / {self.watch["termination"]}', 744, 491, 16, MUTED)
        p.text(self.watch['game_id'], 744, 526, 12, MUTED, True)
        p.text(self.watch['model_id'], 744, 551, 12, MUTED, True)
        app.button('< module', 744, 590, 124, lambda: self.change_module(-1))
        app.button('module >', 878, 590, 124, lambda: self.change_module(1))
        app.button('<', 1020, 590, 42, lambda: setattr(self, 'channel', max(0, self.channel-1)))
        p.text(str(self.channel), 1080, 600, 14, INK, True)
        app.button('>', 1128, 590, 42, lambda: setattr(self, 'channel', self.channel+1))
        app.button('Tensor', 1190, 590, 194, lambda: setattr(self, 'tensor_index', self.tensor_index+1))
        data = self.watch_data
        if SPEEDS[self.speed_index] is None:
            p.text('MAX / activation refresh suppressed.', 744, 665, 21)
        elif data:
            current = data['fen'] == self.watch_game.board.fen()
            p.text(('Current' if current else 'Previous')+' recorded-position sample / '+data['module'], 744, 643, 16)
            items = [v for v in (data['evaluation'].inspection or {}).get('tensors', {}).values() if v['sample_values']]
            heatmap(p, items[self.tensor_index % len(items)] if items else None, 744, 681, 'ACTIVATION / SLICE '+str(self.channel), self.channel)
            p.text(f'Value {data["evaluation"].value:+.4f}', 943, 732, 22, INK, True)
            p.text('Source ply '+str(data['ply']), 943, 780, 15, MUTED, True)
            p.text(('White' if chess.Board(data['fen']).turn else 'Black')+' to move', 943, 758, 14, MUTED)
            p.text('Recomputed using the exact archived model.', 943, 820, 14, MUTED)
        else:
            p.text('Waiting for a sampled forward pass.', 744, 664, 20, MUTED)

    def draw_config(self, app):
        p, button = app.p, app.button
        button('Diagnostics '+('on' if self.telemetry else 'off'), 744, 415, 203, self.toggle_telemetry, active=self.telemetry)
        button('Activations '+('on' if self.activations else 'off'), 957, 415, 213, self.toggle_activations, active=self.activations)
        button('Gradients '+('on' if self.gradients else 'off'), 1180, 415, 204, self.toggle_gradients, active=self.gradients)
        button(f'Every {self.interval} steps', 744, 465, 204, self.change_interval)
        p.text('Next sample: '+self.module[:34], 974, 476, 14, MUTED)
        cfg = self.config
        rows = [('Device / workers', f'{cfg.device} / {cfg.selfplay_workers}'), ('Batch / updates per generation', f'{cfg.batch_size} / {cfg.updates_per_generation}'),
                ('Games / max plies', f'{cfg.games_per_generation} / {cfg.max_game_plies}'), ('MCTS simulations', str(cfg.mcts_simulations)),
                ('Evaluation pairs', str(cfg.evaluation_pairs) if cfg.evaluation_enabled else 'Disabled'), ('Network channels / blocks', f'{cfg.network_channels} / {cfg.residual_blocks}')]
        for i, (label, value) in enumerate(rows):
            y = 527+i*37
            p.text(label, 744, y, 17, MUTED)
            p.text(value, 1228, y, 17, INK, True)
        p.text('Drop a TOML configuration before starting a new run.', 744, 774, 16, MUTED)
        p.text('Resume always restores the saved run configuration.', 744, 809, 16, MUTED)
        p.text('Pause/stop finish the current wave or minibatch; evaluation', 744, 848, 15, MUTED)
        p.text('can stop between search operations. Closing waits for safety.', 744, 875, 15, MUTED)

    def draw_human(self, app):
        p, button = app.p, app.button
        p.text('Learn from your recorded games.', 744, 417, 25)
        button('Save games / '+('on' if app.save_human else 'off'), 744, 467, 240,
            lambda: setattr(app, 'save_human', not app.save_human), active=app.save_human)
        p.text('Opt-in / local PGNs only', 1009, 479, 16, MUTED)
        p.text('Games save when you begin another match or close the app.', 744, 530, 16, MUTED)
        p.text('Only completed human decisions train. Incomplete games stay', 744, 562, 16, MUTED)
        p.text('archived with no outcome target. Undo keeps the surviving line.', 744, 590, 16, MUTED)
        base = self.human_base(app)
        p.label('BASE CHECKPOINT / SELECT IN PLAY SETUP', 744, 638)
        p.text(base.name if base else 'No checkpoint / train self-play or load weights first', 744, 665, 17)
        button('Train from human games', 744, 711, 300, lambda: self.start_human(base), enabled=base is not None and not self.worker.busy)
        p.text('1 epoch / learning rate 0.0001', 1060, 724, 13, MUTED)
        report = self.human_report
        if report:
            evaluation = report.get('evaluation') or {}
            p.text(f'{report["positions"]} positions / {len(report["updates"])} updates / {report["status"]}', 744, 775, 15, INK, True)
            p.text(evaluation.get('reason', 'Separate supervised candidate. Best waits for the evaluation gate.')[:78], 744, 815, 14, MUTED)
            button('Play candidate', 744, 861, 190, lambda: self.play_human_candidate(app), enabled=report['status'] == 'complete')
        else:
            p.text('Supervised move imitation + outcome learning; it may learn', 744, 785, 16, MUTED)
            p.text('your mistakes. Promotion still requires the same match gate.', 744, 816, 16, MUTED)
            p.text('Human data never enters the self-play replay database.', 744, 863, 15, MUTED)

    def play_human_candidate(self, app):
        path = Path(self.human_report['candidate'])
        if path not in app.models:
            app.models.append(path)
        app.weights = path
        app.new_game()

    def cycle_generations(self):
        values = (1, 2, 5, 10)
        self.generations = values[(values.index(self.generations)+1) % len(values)]

    def manual_step(self, amount):
        self.watch_playing = False
        self.step_watch(amount)

    def request_close(self):
        self.closing = True
        self.worker.request('stop')
        self.brain.stop()

    def close(self):
        self.brain.close()
        self.worker.cleanup()
