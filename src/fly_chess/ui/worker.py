"""One owning thread, one outstanding job, bounded detached GUI messages."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Event, Lock

_MODEL_LOAD_LOCK = Lock()

DIFFICULTIES = (("Easy", 16, 1., .5), ("Medium", 64, 2., .15),
                ("Hard", 128, 3., 0.), ("Full power", 256, 5., 0.))


class BrainWorker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-brain')
        self.messages = Queue(maxsize=4)
        self.future = None
        self.cancel = Event()
        self.evaluator = None
        self.model_key = None

    @property
    def busy(self):
        return self.future is not None and not self.future.done()

    def stop(self):
        self.cancel.set()

    def close(self):
        self.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def publish(self, message):
        while True:
            try:
                self.messages.put_nowait(message)
                return
            except Full:
                try:
                    self.messages.get_nowait()
                except Empty:
                    pass

    def drain(self):
        rows = []
        while True:
            try:
                rows.append(self.messages.get_nowait())
            except Empty:
                return rows

    def submit(self, *, token, game, human_color, weights=None, difficulty=1,
               adapt=False, module='stem', search=False, telemetry=True):
        if self.busy:
            return False
        self.cancel = Event()
        self.future = self.executor.submit(self._run, token, game.copy(), human_color,
            weights, difficulty, adapt, module, search, telemetry, self.cancel)
        return True

    def _run(self, token, game, human_color, weights, difficulty, adapt, module, search, telemetry, cancel):
        try:
            import numpy as np
            from fly_chess.core.encoding import encode_board, PLANE_NAMES
            from fly_chess.core.actions import decode_action
            from fly_chess.game.opponent import rebuild_profile, OpponentSnapshot
            from fly_chess.neural.runtime import seed_everything, select_device
            from fly_chess.neural.weights import load_weights
            from fly_chess.neural.network import PolicyValueNetwork, module_inventory
            from fly_chess.neural.inference import Evaluator
            from fly_chess.neural.inspection import InspectionRequest
            from fly_chess.search.mcts import MCTS, SearchSettings
            model_key = (weights, token[0])
            if self.evaluator is None or self.model_key != model_key:
                # Freeze weights within a match; a new match reloads a replaced latest/best file.
                # Two GUI inference owners must not interleave seeded model construction.
                with _MODEL_LOAD_LOCK:
                    seed_everything(42, cpu_threads=2)
                    device = select_device('auto').device
                    model, name = load_weights(weights, device=device) if weights else (PolicyValueNetwork().to(device), 'Untrained / exploration only')
                    self.evaluator = Evaluator(model, model_id=name)
                    self.model_key = model_key
            if cancel.is_set():
                return
            evaluator = self.evaluator
            profile = rebuild_profile(game, human_color, evaluator, cancel.is_set) if adapt else OpponentSnapshot()
            if cancel.is_set():
                return
            inspection = InspectionRequest(str(token), max_values=65536, values_per_tensor=65536,
                selected_module=module) if telemetry else None
            evaluation = evaluator.evaluate([game], inspection=inspection)[0]
            payload = {'token': token, 'kind': 'inspection', 'fen': game.board.fen(),
                'ply': len(game.board.move_stack), 'evaluation': evaluation,
                'inventory': module_inventory(evaluator.model), 'profile': profile.as_dict(),
                'module': module, 'weights': [], 'input': None, 'reply_predictions': []}
            if adapt and game.board.turn == human_color and not evaluation.terminal:
                actions = game.actions()
                baseline = evaluation.policy[actions]
                prediction = profile.predict(game, actions, baseline)
                indices = np.argsort(-prediction)[:3]
                payload['reply_predictions'] = [{'move': decode_action(int(actions[i])).uci(),
                    'baseline': float(baseline[i]), 'adapted': float(prediction[i])} for i in indices]
            if telemetry:
                payload['input'] = {'names': PLANE_NAMES, 'values': encode_board(game.board).tolist()}
                modules = dict(evaluator.model.named_modules())
                selected = modules.get('' if module == '<network>' else module)
                if selected is not None:
                    payload['weights'] = [{'name': name, 'shape': list(parameter.shape),
                        'total': parameter.numel(), 'values': parameter.detach().reshape(-1)[:4096].float().cpu().tolist()}
                        for name, parameter in selected.named_parameters(recurse=False)]
            if not cancel.is_set():
                self.publish(payload)
            if search and not cancel.is_set():
                _, simulations, seconds, temperature = DIFFICULTIES[difficulty]
                settings = SearchSettings(simulations=simulations, max_seconds=seconds,
                    max_nodes=4096, opponent_weight=profile.strength if adapt else 0.)
                def progress(snapshot):
                    if telemetry and not cancel.is_set():
                        self.publish({**payload, 'kind': 'progress', 'search': snapshot})
                result = MCTS(evaluator, settings, seed=42+len(game.board.move_stack)).search(game,
                    temperature=temperature, cancel=cancel.is_set, request_id=str(token), game_id=token[0],
                    tree_edges=512 if telemetry else 0,
                    opponent_prior=profile.predict if adapt and profile.strength else None,
                    on_progress=progress if telemetry else None)
                if not cancel.is_set():
                    self.publish({**payload, 'kind': 'result', 'search': result.snapshot, 'move': result.move})
        except Exception as exc:
            if not cancel.is_set():
                self.publish({'token': token, 'kind': 'error', 'error': f'{type(exc).__name__}: {exc}'})
