"""Board-first monochrome interface. All inference runs outside the event loop."""
from pathlib import Path
import time
import chess
import pygame as pg

from fly_chess.game.session import HumanSession
from fly_chess.ui.board import BOARD, draw_board, square_at, square_rect
from fly_chess.ui.theme import BG, INK, MUTED, RULE, SIZE, Painter, piece_image
from fly_chess.ui.worker import BrainWorker, DIFFICULTIES


class Application:
    def __init__(self, workspace, weights=None, config=None):
        self.workspace = Path(workspace)
        self.models = sorted((self.workspace / 'models').glob('*.pt'))
        if weights and Path(weights) not in self.models:
            self.models.insert(0, Path(weights))
        self.weights = Path(weights) if weights else (self.models[0] if self.models else None)
        self.match_weights = self.weights
        self.match_model_id = 'unavailable / no forward pass yet'
        self.match_adaptation = False
        self.save_human = False
        from fly_chess.game.human_data import HumanArchiveWorker
        self.archives = HumanArchiveWorker()
        self._archived_token = None
        self.screen = 'menu'
        self.session = HumanSession()
        self.human_color = chess.WHITE
        self.difficulty = 1
        self.adapt = False
        self.reduced_motion = False
        self.telemetry = True
        self.flipped = False
        self.selected = None
        self.cursor = chess.E2
        self.promotion = []
        self.tab = 'Brain'
        from fly_chess.ui.brain_view import BrainView
        self.brain_view = BrainView()
        self.module = 'stem'
        self.channel = 0
        self.layer_mode = 'Activations'
        self.tensor_index = 0
        self.tree_page = 0
        self.root_page = 0
        self.candidate = None
        self.data = None
        self.decisions = []
        self.frozen = False
        self.frozen_index = -1
        self.frozen_data = None
        self.error = ''
        self.animation = None
        self.worker = BrainWorker()
        from fly_chess.ui.training_view import TrainingView
        self.training = TrainingView(self.workspace, config)
        self._last_training_state = None
        self.closing = False
        self.requested = None
        self.buttons = []
        self.focus = -1
        self.mouse = (-1, -1)
        self.canvas = pg.Surface(SIZE)
        self.p = Painter(self.canvas)
        self.running = True

    def button(self, label, x, y, w, action, enabled=True, active=False):
        rect = pg.Rect(x, y, w, 38)
        index = len(self.buttons)
        self.buttons.append((rect, action, enabled))
        hover = rect.collidepoint(self.mouse) and enabled
        self.p.rect(INK if active else (36, 36, 35) if hover else BG, rect)
        self.p.rect(INK if active or hover else RULE, rect, 1)
        if self.focus == index:
            self.p.rect(INK, rect.inflate(6, 6), 1)
        self.p.text(label, x+12, y+8, 15, BG if active else INK if enabled else RULE)

    def header(self):
        p = self.p
        p.text('FLY / CHESS', 48, 30, 24)
        p.label('A learning machine. A game of decisions.', 258, 39)
        self.button('Play', 1054, 27, 94, lambda: self.navigate('menu'), active=self.screen in ('menu', 'play'))
        self.button('Train', 1158, 27, 94, lambda: self.navigate('train'), active=self.screen == 'train')
        self.button('Settings', 1262, 27, 130, lambda: self.navigate('settings'), active=self.screen == 'settings')
        p.line(48, 83, 1344)

    def draw(self, scale=1):
        target = (round(SIZE[0]*scale), round(SIZE[1]*scale))
        if self.canvas.get_size() != target or self.p.scale != scale:
            self.canvas = pg.Surface(target)
            self.p = Painter(self.canvas, scale)
        self.buttons = []
        self.canvas.fill(BG)
        self.header()
        if self.screen == 'menu':
            self.draw_menu()
        elif self.screen == 'play':
            self.draw_play()
        elif self.screen == 'train':
            self.training.draw(self)
        else:
            self.draw_settings()
        self.p.line(48, 916, 1344)
        self.p.text('FLY  /  LOCAL LAB', 48, 934, 12, MUTED, True)
        self.p.text('Tab / Enter: controls    Arrows / Enter: board    Esc: cancel', 738, 934, 12, MUTED, True)
        if self.error and (self.screen != 'train' or self.archives.failed):
            self.p.rect(INK, (48, 865, 1344, 44))
            self.p.text('ERROR  '+self.error[:123], 60, 875, 14, BG, True)
        if self.archives.failed:
            self.button('Retry save', 1250, 865, 142, self.retry_archives)
        return self.canvas

    def draw_menu(self):
        p = self.p
        p.label('01 / PLAY A MATCH', 56, 119)
        p.text('Every move leaves a trace.', 56, 143, 32)
        draw_board(p, self.session.game.board, self.flipped)
        p.text('Your board. Its thinking, visible.', 56, 838, 22)
        p.label('MATCH SETUP', 744, 130)
        p.text('Meet Fly across the board.', 744, 158, 30)
        p.text('Inspect the network and search behind each move.', 744, 209, 18, MUTED)
        p.line(744, 254, 640)
        p.label('01   MODEL', 744, 278)
        model = self.weights.name if self.weights else 'Untrained / exploration only' if self.models else 'No trained checkpoint found'
        p.text(model[:48], 744, 309, 21)
        p.text('Strength is unmeasured.' if self.weights else 'Start training, or explore an untrained network.', 744, 346, 16, MUTED)
        self.button('Next model', 744, 381, 140, self.next_model, enabled=bool(self.models))
        p.text('Drop a .pt checkpoint anywhere to load it.', 906, 391, 14, MUTED)
        p.line(744, 444, 640)
        p.label('02   YOUR SIDE', 744, 468)
        for i, (name, color) in enumerate((('White', True), ('Black', False))):
            self.button(name, 744+i*126, 498, 116, lambda c=color: self.set_side(c), active=self.human_color == color)
        p.label('03   SEARCH BUDGET', 744, 566)
        for i, item in enumerate(DIFFICULTIES):
            self.button(item[0], 744+i*158, 597, 148, lambda n=i: setattr(self, 'difficulty', n), active=self.difficulty == i)
        _, simulations, seconds, _ = DIFFICULTIES[self.difficulty]
        p.text(f'Up to {simulations} simulations / {seconds:g}s search / 4,096 nodes', 744, 652, 15, MUTED, True)
        self.button(('On' if self.adapt else 'Off')+' / Adapt to me', 744, 703, 218, self.toggle_adapt, active=self.adapt)
        p.text('Experimental. Match memory only.', 982, 713, 14, MUTED)
        self.button('Begin match' if self.weights else 'Explore untrained Fly', 744, 790, 300, self.new_game, active=True)
        p.text('Weights remain unchanged during play.', 744, 848, 15, MUTED)

    def draw_play(self):
        p = self.p
        p.label('01 / HUMAN VS FLY', 56, 112)
        result = self.session.game.result
        status = self.session.manual_result or (f'{result.termination.replace("_", " ")} / {self.session.game.board.result()}' if result else 'Your move' if self.session.human_turn else 'Fly is thinking')
        p.text(status, 56, 139, 29)
        p.text('WHITE' if self.session.game.board.turn else 'BLACK', 582, 151, 14, MUTED, True)
        animation = None
        if self.animation:
            move, piece, start = self.animation
            progress = (time.monotonic()-start)/.16
            if progress < 1 and not self.reduced_motion:
                animation = (move, piece, max(0, progress))
            else:
                self.animation = None
        draw_board(p, self.session.game.board, self.flipped, self.selected, self.cursor, animation)
        if self.candidate and self.candidate[0] == self.session.game.board.fen():
            move = self.candidate[1]
            a, b = square_rect(move.from_square, self.flipped).center, square_rect(move.to_square, self.flipped).center
            self.p.segment(BG, a, b, 7)
            self.p.segment(INK, a, b, 3)
            self.p.circle(INK, b, 11, 3)
        self.button('Undo', 56, 839, 84, self.undo, enabled=self.session.can_undo)
        self.button('Flip', 150, 839, 74, lambda: setattr(self, 'flipped', not self.flipped))
        self.button('New', 234, 839, 78, self.new_game)
        self.button('Resign', 322, 839, 96, self.resign, enabled=not self.session.finished)
        self.button('Claim draw', 428, 839, 132, self.claim_draw,
                    enabled=self.session.human_turn and self.session.game.board.can_claim_draw())
        self.button('Menu', 570, 839, 94, lambda: self.navigate('menu'))
        self.draw_brain()
        if self.promotion:
            self.p.rect(BG, (146, 433, 426, 140))
            p.text('Choose promotion', 164, 442, 18)
            for i, move in enumerate(self.promotion):
                x = 164+i*99
                self.button(chess.piece_name(move.promotion).title(), x, 521, 93, lambda m=move: self.play_move(m))
                self.p.piece(chess.Piece(move.promotion, self.human_color), 50, (x+19, 469))

    def draw_brain(self):
        p = self.p
        p.label('FLY BRAIN / INFERENCE ONLY', 744, 112)
        self.button('Live' if self.frozen else 'Freeze', 1290, 101, 94, self.toggle_freeze, enabled=bool(self.data))
        self.button('<', 1190, 101, 38, lambda: self.step_record(-1), enabled=bool(self.decisions))
        self.button('>', 1238, 101, 38, lambda: self.step_record(1), enabled=bool(self.decisions))
        data = self.visible_data()
        label = 'Unavailable / awaiting worker'
        if data:
            current = data['token'] == self.session.token
            label = ('Frozen' if self.frozen else 'Current sample' if current else 'Recorded decision')+f' / ply {data["ply"]} / '+('White' if chess.Board(data['fen']).turn else 'Black')+' to move'
        p.text(label, 744, 151, 15, MUTED, True)
        for i, tab in enumerate(('Brain', 'Overview', 'Layers', 'Search', 'Opponent')):
            self.button(tab, 744+i*130, 190, 120, lambda t=tab: setattr(self, 'tab', t), active=tab == self.tab)
        if not data:
            p.text('Waiting for a real forward pass.', 744, 280, 24)
            p.text('The board remains interactive while Fly loads.', 744, 323, 17, MUTED)
            return
        ev = data['evaluation']
        p.text(ev.model_id[:64], 744, 243, 12, MUTED, True)
        p.text('Position '+ev.position_id[:16], 744, 267, 12, MUTED, True)
        if self.tab == 'Brain':
            self.button('Reset view', 1244, 285, 140, self.brain_view.reset)
            p.text('Scroll to zoom / drag to rotate / hover to inspect', 744, 298, 13, MUTED)
            self.brain_view.draw(p, ev.inspection, self.mouse, y=330, height=350)
            snap = ev.inspection or {}
            p.text('Captured '+snap.get('timestamp_utc', 'unavailable')[11:23]+' UTC / latest position sample', 744, 750, 12, MUTED, True)
        elif self.tab == 'Overview':
            self.draw_overview(data)
        elif self.tab == 'Layers':
            self.draw_layers(data)
        elif self.tab == 'Search':
            self.draw_search(data)
        else:
            self.draw_opponent(data)
        p.line(744, 780, 640)
        p.label('MOVE RECORD / SAN', 744, 796)
        history = self.session.history()
        rows = [f'{i//2+1}. {history[i]}'+('  '+history[i+1] if i+1 < len(history) else '') for i in range(0, len(history), 2)]
        p.text('   '.join(rows[-6:-3])[:72] or ('No moves yet.' if not rows else ''), 744, 822, 15, INK, True)
        p.text('   '.join(rows[-3:])[:72], 744, 846, 15, INK, True)
        captured = self.session.captured()
        p.text('Captured by you: '+(' '.join(chess.piece_symbol(t).upper() for t in captured[self.human_color]) or '—'), 744, 877, 13, MUTED, True)

    def draw_overview(self, data):
        p = self.p
        p.label('DECISION PATH / ACTUAL NETWORK', 744, 311)
        steps = [('01', '21 board planes', 'pieces, turn, castling, history'),
                 ('02', 'Stem + residual trunk', f'{len(data["inventory"])} inspectable modules'),
                 ('03', 'Policy + value heads', '20,480 actions / scalar outcome'),
                 ('04', 'Monte Carlo tree search', 'legal priors → visits → move')]
        for i, (n, title, detail) in enumerate(steps):
            y = 348+i*72
            p.text(n, 744, y, 18, MUTED, True)
            p.text(title, 795, y-3, 21)
            p.text(detail, 795, y+27, 14, MUTED)
            p.line(744, y+61, 640)
        value = data['evaluation'].value
        p.label('VALUE / PLAYER TO MOVE / NOT WIN PROBABILITY', 744, 649)
        p.text(f'{value:+.3f}', 744, 678, 36, INK, True)
        self.p.rect(RULE, (940, 698, 444, 5))
        self.p.circle(INK, (int(940+(value+1)*222), 700), 7)
        p.text('−1 loss', 940, 717, 13, MUTED)
        p.text('+1 win', 1333, 717, 13, MUTED)
        snap = data['evaluation'].inspection
        p.text(('Captured '+snap['timestamp_utc'][11:23]+' UTC') if snap else 'Telemetry disabled / no activation capture', 744, 752, 12, MUTED, True)

    def draw_layers(self, data):
        p = self.p
        self.button('< module', 744, 305, 120, lambda: self.change_module(-1), enabled=not self.frozen)
        self.button('module >', 874, 305, 120, lambda: self.change_module(1), enabled=not self.frozen)
        self.button(self.layer_mode, 1010, 305, 156, self.cycle_layer_mode)
        self.button('<', 1244, 305, 42, lambda: self.shift_channel(-1))
        self.button('>', 1342, 305, 42, lambda: self.shift_channel(1))
        p.text(str(self.channel), 1300, 315, 14, INK, True)
        p.text(data['module'][:52], 744, 364, 22)
        module = next((m for m in data['inventory'] if m['name'] == data['module']), {})
        p.text(f'{module.get("type", "")} / {module.get("parameters", 0):,} own parameters', 744, 400, 15, MUTED, True)
        self.button('Tensor '+str(self.tensor_index+1), 1240, 385, 144, self.next_tensor)
        tensors = (data['evaluation'].inspection or {}).get('tensors', {})
        items = [(k, v) for k, v in tensors.items() if v['sample_values']]
        values, detail = [], 'No tensor available.'
        if self.layer_mode == 'Input' and data['input']:
            index = self.channel % len(data['input']['values'])
            values = sum(data['input']['values'][index], [])
            detail = f'Plane {index}: {data["input"]["names"][index]} / a1 bottom left'
        elif self.layer_mode == 'Weights' and data['weights']:
            item = data['weights'][self.tensor_index % len(data['weights'])]
            values = item['values'][self.channel*64:(self.channel+1)*64]
            detail = f'{item["name"]} {item["shape"]} / slice {self.channel*64}:{self.channel*64+len(values)} of {item["total"]}'
        elif self.layer_mode == 'Activations' and items:
            key, item = items[self.tensor_index % len(items)]
            values = item['sample_values'][self.channel*64:(self.channel+1)*64]
            spatial = item['shape'][-2:] == [8, 8]
            detail = f'{item["shape"]} / '+('channel ' if spatial else 'flat slice ')+str(self.channel)
        if not values:
            p.text('No values in this slice.', 744, 460, 21)
            p.text('Try channel 0, another module, or enable telemetry.', 744, 503, 16, MUTED)
            return
        p.text(detail[:75], 744, 434, 13, MUTED, True)
        scale = max(max(abs(v) for v in values), 1e-8)
        for i, value in enumerate(values):
            row, col = divmod(i, 8)
            gray = int(127+120*value/scale)
            self.p.rect((gray, gray, gray), (744+col*30, 476+(7-row)*30, 28, 28))
        p.label('PER-SLICE SYMMETRIC SCALE', 1020, 479)
        p.text(f'−{scale:.4g}  …  +{scale:.4g}', 1020, 511, 17, INK, True)
        p.text(f'Min  {min(values):+.5f}', 1020, 558, 16, MUTED, True)
        p.text(f'Max  {max(values):+.5f}', 1020, 587, 16, MUTED, True)
        p.text(f'Mean {sum(values)/len(values):+.5f}', 1020, 616, 16, MUTED, True)
        bins = [0]*16
        for value in values:
            bins[min(15, max(0, int((value/scale+1)*8)))] += 1
        for i, count in enumerate(bins):
            height = round(40*count/max(bins))
            self.p.rect(MUTED, (1020+i*20, 689-height, 17, height))
        p.text('Slice distribution / −scale … +scale', 1020, 700, 13, MUTED)
        p.text('All modules reachable with < module >. Empty slices are explicit.', 744, 744, 13, MUTED)

    def draw_search(self, data):
        p = self.p
        search = data.get('search')
        if not search:
            from fly_chess.core.actions import encode_move
            board = chess.Board(data['fen'])
            rows = [{'move': m.uci(), 'visits': 0, 'q': 0., 'network_prior': float(data['evaluation'].policy[encode_move(m)])}
                    for m in board.legal_moves]
            rows.sort(key=lambda row: -row['network_prior'])
            search = {'root_moves': rows, 'tree': [], 'simulations': 0, 'allocated_nodes': 0,
                      'stop_reason': 'Network policy only', 'elapsed_seconds': 0., 'principal_variation': []}
        p.text(f'{search["simulations"]} simulations / {search["allocated_nodes"]} nodes', 744, 310, 22)
        p.text(f'{search["stop_reason"]} / {search["elapsed_seconds"]:.2f}s', 744, 346, 14, MUTED, True)
        self.button('Root / tree' if not self.tree_page else 'Tree / root', 1224, 342, 160, self.toggle_tree)
        p.label('MOVE     VISITS  SHARE  PRIOR    Q (PARENT)', 744, 401)
        if not self.tree_page:
            start = self.root_page*7
            rows = search['root_moves'][start:start+7]
        else:
            start = (self.tree_page-1)*7
            rows = search['tree'][start:start+7]
        for i, row in enumerate(rows):
            y = 434+i*31
            chosen = row['move'] == search.get('selected_move') and not self.tree_page
            share = row['visits']/max(1, search['simulations'])
            text = f'{">" if chosen else " "}{row["move"]:6} {row["visits"]:5}  {share:5.1%} {row["network_prior"]:5.1%}  {row["q"]:+.3f}'
            p.text(text, 744, y, 14, INK, True)
            if not self.tree_page:
                self.buttons.append((pg.Rect(744, y, 640, 29),
                    lambda r=row: setattr(self, 'candidate', (data['fen'], chess.Move.from_uci(r['move']))), True))
            if self.tree_page:
                p.text('/'.join(row['path'])[-20:] or 'root', 1200, y, 12, MUTED, True)
        pv = ' '.join(search['principal_variation'])
        p.text('Line  '+pv[:64], 744, 665, 13, MUTED, True)
        p.text('Q uses parent-player perspective; unvisited Q is zero.', 744, 695, 14, MUTED)
        if self.tree_page:
            self.button('Next 7 edges', 744, 726, 170, lambda: self.next_tree(search))
            p.text('Bounded tree / '+str(len(search['tree']))+' edges captured', 940, 737, 13, MUTED)
        else:
            self.button('Next candidates', 744, 726, 184,
                lambda: setattr(self, 'root_page', (self.root_page+1) % max(1, (len(search['root_moves'])+6)//7)))
            p.text('Click move / overlay on matching board only.', 943, 738, 12, MUTED)

    def draw_opponent(self, data):
        p = self.p
        profile = data['profile']
        p.text('Adapt to me / '+('On' if self.adapt else 'Off'), 744, 315, 25)
        p.text('Experimental. No strength benefit established.', 744, 356, 16, MUTED)
        p.line(744, 393, 640)
        p.text(profile['evidence'], 744, 418, 22)
        p.text(f'{profile["decisions"]} observed decisions / {profile["informative"]} informative', 744, 459, 15, MUTED, True)
        for i, (name, weight) in enumerate(zip(profile['features'], profile['weights'])):
            y = 512+i*48
            p.text(name, 744, y, 18)
            p.text(f'{weight:+.3f}', 1280, y, 18, INK, True)
            self.p.segment(RULE, (1070, y+13), (1210, y+13), 2)
            self.p.circle(INK, (int(1140+weight*35), y+13), 4)
        p.text(f'Search-prior blend: {profile["influence"]:.1%} / cap below 20%', 744, 655, 15, MUTED, True)
        predictions = data['reply_predictions']
        if predictions:
            p.label('HUMAN REPLIES / BASELINE → ADAPTED', 744, 692)
            p.text('   '.join(f'{r["move"]} {r["baseline"]:.0%}→{r["adapted"]:.0%}' for r in predictions), 744, 720, 14, INK, True)
        else:
            p.text('Reply comparison available on your turn with adaptation on.', 744, 712, 14, MUTED)
        p.text('Undo rebuilds memory. New match clears it. Weights stay fixed.', 744, 749, 14, MUTED)

    def draw_settings(self):
        p = self.p
        p.label('03 / SETTINGS', 56, 122)
        p.text('A quieter instrument.', 56, 166, 38)
        options = [('Reduced motion', self.reduced_motion, lambda: setattr(self, 'reduced_motion', not self.reduced_motion), 'Skip the short piece-movement transition.'),
                   ('Brain telemetry', self.telemetry, self.toggle_telemetry, 'Capture bounded real activations and search snapshots.'),
                   ('Adapt to me', self.adapt, self.toggle_adapt, 'Experimental, temporary opponent model. Default off.'),
                   ('Save human games', self.save_human, lambda: setattr(self, 'save_human', not self.save_human), 'Opt-in local PGNs on New or Exit. Training requires a separate explicit action.')]
        for i, (label, value, action, description) in enumerate(options):
            y = 265+i*115
            p.line(56, y-22, 1328)
            p.text(label, 56, y, 26)
            p.text(description, 56, y+45, 18, MUTED)
            self.button('On' if value else 'Off', 1230, y, 150, action, active=value)
        p.text('Window: resizable / minimum 1200 × 800 / scaled from 1440 × 960', 56, 734, 17, MUTED)
        p.text('Search: one worker / at most 256 simulations, 4,096 nodes and 5 seconds*', 56, 774, 17, MUTED)
        p.text('*An in-flight neural evaluation may finish beyond the soft time limit.', 56, 814, 15, MUTED)
        self.button('Return to match', 56, 865, 200, lambda: self.navigate('play'))

    def visible_data(self):
        if self.frozen:
            return self.frozen_data
        return self.data

    def navigate(self, screen):
        self.screen = screen
        self.focus = -1

    def next_model(self):
        if self.models:
            choices = [None, *self.models]
            self.weights = choices[(choices.index(self.weights)+1) % len(choices)]

    def set_side(self, color):
        self.human_color = color

    def toggle_adapt(self):
        self.adapt = not self.adapt
        self.invalidate()

    def toggle_telemetry(self):
        self.telemetry = not self.telemetry
        self.invalidate()

    def invalidate(self):
        self.worker.stop()
        self.session.invalidate()
        self.requested = None
        self.error = ''

    def new_game(self):
        if not self.archive_current():
            return
        self.worker.stop()
        self.session = HumanSession(self.human_color)
        self.match_weights = self.weights
        self.match_model_id = 'unavailable / no forward pass yet'
        self.match_adaptation = self.adapt
        self.flipped = not self.human_color
        self.selected = None
        self.promotion = []
        self.data = None
        self.decisions = []
        self.frozen = False
        self.frozen_index = -1
        self.frozen_data = None
        self.requested = None
        self.error = ''
        self.animation = None
        self.candidate = None
        self.module = 'stem'
        self.root_page = self.tree_page = self.channel = self.tensor_index = 0
        self.navigate('play')

    def play_move(self, move):
        if not self.session.human_turn or move not in self.session.game.board.legal_moves:
            return
        piece = self.session.game.board.piece_at(move.from_square)
        self.worker.stop()
        self.session.move_human(move)
        self.animation = (move, piece, time.monotonic())
        self.selected = None
        self.promotion = []
        self.requested = None

    def undo(self):
        if self.session.undo():
            self.worker.stop()
            self.requested = None
            self.selected = None
            self.promotion = []
            self.animation = None
            self.frozen = False
            self.data = None
            ply = len(self.session.game.board.move_stack)
            self.decisions = [d for d in self.decisions if d['ply'] < ply]

    def resign(self):
        self.worker.stop()
        self.session.resign()
        self.promotion = []
        self.selected = None

    def claim_draw(self):
        if self.session.claim_draw():
            self.worker.stop()

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.frozen_index = -1
        self.frozen_data = self.data if self.frozen else None

    def step_record(self, delta):
        if not self.decisions:
            return
        index = self.frozen_index if self.frozen and self.frozen_index >= 0 else len(self.decisions)
        self.frozen_index = max(0, min(len(self.decisions)-1, index+delta))
        self.frozen = True
        self.frozen_data = self.decisions[self.frozen_index]

    def change_module(self, delta):
        if not self.data or self.frozen:
            return
        names = [m['name'] for m in self.data['inventory']]
        self.module = names[(names.index(self.module)+delta) % len(names)]
        self.channel = 0
        self.invalidate()

    def cycle_layer_mode(self):
        modes = ('Activations', 'Weights', 'Input')
        self.layer_mode = modes[(modes.index(self.layer_mode)+1) % len(modes)]
        self.channel = 0
        self.tensor_index = 0

    def next_tensor(self):
        self.tensor_index += 1
        self.channel = 0

    def shift_channel(self, delta):
        self.channel = max(0, self.channel+delta)

    def toggle_tree(self):
        self.tree_page = 0 if self.tree_page else 1

    def next_tree(self, search):
        self.tree_page = self.tree_page % max(1, (len(search['tree'])+6)//7)+1

    def board_click(self, square):
        if square is None or not self.session.human_turn or self.promotion:
            return
        board = self.session.game.board
        choices = [m for m in board.legal_moves if m.from_square == self.selected and m.to_square == square]
        if len(choices) > 1:
            self.promotion = sorted(choices, key=lambda m: -m.promotion)
        elif choices:
            self.play_move(choices[0])
        else:
            piece = board.piece_at(square)
            self.selected = square if piece and piece.color == self.session.human_color else None

    def tick(self):
        errors = self.archives.drain()
        if errors:
            self.error = errors[-1]
            self.closing = False
            self.training.closing = False
        self.training.tick()
        if self.training.state is not self._last_training_state:
            self._last_training_state = self.training.state
            self.models = sorted(set(self.models) | set((self.workspace / 'models').glob('*.pt')))
        if self.closing and not self.training.worker.busy and not self.archives.pending and not self.archives.failed:
            self.running = False
        for message in self.worker.drain():
            if message['token'] != self.session.token:
                continue
            if message['kind'] == 'error':
                self.error = message['error']
                continue
            self.data = message
            self.match_model_id = message['evaluation'].model_id
            if message['kind'] == 'result':
                self.decisions.append(message)
                self.decisions = self.decisions[-16:]
                if message['move'] is not None:
                    piece = self.session.game.board.piece_at(message['move'].from_square)
                    if self.session.apply_ai(message['token'], message['move']):
                        self.animation = (message['move'], piece, time.monotonic())
                        self.requested = None
                elif not self.session.finished:
                    self.error = 'Search returned no move. Choose New to retry with a larger search budget.'
        if self.screen == 'play' and not self.closing and not self.session.finished and self.requested != self.session.token and not self.error:
            self.match_adaptation |= self.adapt
            if self.worker.submit(token=self.session.token, game=self.session.game, human_color=self.session.human_color,
                    weights=self.match_weights, difficulty=self.difficulty, adapt=self.adapt, module=self.module,
                    search=not self.session.human_turn, telemetry=self.telemetry):
                self.requested = self.session.token

    def handle_event(self, event, position=None):
        if self.closing:
            return
        viewer = self.brain_view if self.screen == 'play' and self.tab == 'Brain' else self.training.brain_view if self.screen == 'train' and self.training.pane == 'Brain' else None
        if viewer and viewer.handle(event, position if position is not None else self.mouse):
            return
        if event.type == pg.QUIT:
            self.worker.stop()
            self.session.invalidate()
            self.requested = None
            if not self.archive_current():
                return
            self.closing = True
            if self.training.worker.busy or self.archives.pending:
                self.training.request_close()
                self.training.notice = 'Closing / waiting for training to save at a safe boundary.'
                if self.training.worker.busy:
                    self.screen = 'train'
            else:
                self.running = False
        elif event.type == pg.DROPFILE:
            path = Path(event.file)
            if self.screen == 'train':
                self.training.drop(path)
                return
            if path.suffix.lower() != '.pt':
                self.error = 'Drop a .pt checkpoint file.'
            else:
                if path not in self.models:
                    self.models.append(path)
                self.weights = path
                self.new_game()
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.focus = -1
            for rect, action, enabled in self.buttons:
                if enabled and rect.collidepoint(position):
                    action()
                    return
            if self.screen == 'play':
                self.board_click(square_at(position, self.flipped))
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_TAB:
                step = -1 if event.mod & pg.KMOD_SHIFT else 1
                for _ in self.buttons:
                    self.focus = (self.focus+step) % len(self.buttons)
                    if self.buttons[self.focus][2]:
                        break
            elif event.key == pg.K_ESCAPE:
                self.promotion = []
                self.selected = None
                self.focus = -1
            elif event.key in (pg.K_RETURN, pg.K_SPACE):
                if 0 <= self.focus < len(self.buttons) and self.buttons[self.focus][2]:
                    self.buttons[self.focus][1]()
                elif self.screen == 'play':
                    self.board_click(self.cursor)
            elif self.screen == 'play' and event.key in (pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN):
                self.focus = -1
                dx, dy = {pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0), pg.K_UP: (0, 1), pg.K_DOWN: (0, -1)}[event.key]
                sign = -1 if self.flipped else 1
                self.cursor = chess.square(max(0, min(7, chess.square_file(self.cursor)+dx*sign)),
                                           max(0, min(7, chess.square_rank(self.cursor)+dy*sign)))

    def archive_current(self):
        if self.archives.failed:
            self.error = 'A human-game save failed. Use Retry save before leaving the match.'
            return False
        if not self.save_human or not self.session.game.board.move_stack:
            return True

        if self._archived_token == self.session.token:
            return True
        try:
            self.archives.submit(self.workspace, self.session, consent=True,
                model_id=self.match_model_id, checkpoint=str(self.match_weights or 'untrained'), adaptation=self.match_adaptation)
            self._archived_token = self.session.token
            return True
        except RuntimeError as exc:
            self.error = str(exc)
            return False

    def retry_archives(self):
        self.archives.retry()
        self.error = ''

    def close(self):
        self.worker.close()
        self.training.request_close()
        while self.training.worker.busy:
            self.training.tick()
            time.sleep(.05)
        self.training.close()
        self.archive_current()
        errors = self.archives.close()
        if errors or self.archives.failed:
            raise RuntimeError(errors[-1] if errors else 'Human-game saves remain failed; their data was not persisted.')


def run(workspace, weights=None, config=None):
    # Set awareness before SDL creates its window so Windows does not bitmap-scale it.
    import sys
    if sys.platform == 'win32':
        import ctypes
        try:
            if not ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    pg.init()
    window = pg.display.set_mode(SIZE, pg.RESIZABLE)
    pg.display.set_caption('Fly / Chess')
    app = Application(workspace, weights, config)
    clock = pg.time.Clock()
    try:
        while app.running:
            width, height = window.get_size()
            scale = min(width/SIZE[0], height/SIZE[1])
            offset = ((width-SIZE[0]*scale)/2, (height-SIZE[1]*scale)/2)
            convert = lambda pos: ((pos[0]-offset[0])/scale, (pos[1]-offset[1])/scale)
            app.mouse = convert(pg.mouse.get_pos())
            for event in pg.event.get():
                if event.type == pg.VIDEORESIZE:
                    window = pg.display.set_mode((max(1200, event.w), max(800, event.h)), pg.RESIZABLE)
                    width, height = window.get_size()
                    scale = min(width/SIZE[0], height/SIZE[1])
                    offset = ((width-SIZE[0]*scale)/2, (height-SIZE[1]*scale)/2)
                else:
                    app.handle_event(event, convert(event.pos) if hasattr(event, 'pos') else None)
            app.mouse = convert(pg.mouse.get_pos())
            app.tick()
            window.fill(BG)
            window.blit(app.draw(scale), (round(offset[0]), round(offset[1])))
            pg.display.flip()
            clock.tick(60)
    finally:
        app.close()
        pg.quit()


if __name__ == '__main__':
    run(Path.cwd())
