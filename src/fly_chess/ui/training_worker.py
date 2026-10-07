"""Spawned training owner: GUI commands are events, reports are bounded copies."""
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time


def offer(queue, message):
    # A slow viewer may lose intermediate reports; training never waits for it.
    try:
        queue.put_nowait(message)
    except Full:
        try:
            queue.get_nowait()
            queue.put_nowait(message)
        except (Empty, Full):
            pass


def watch_record(root, progress):
    from fly_chess.storage.games import load_game
    from fly_chess.core.actions import decode_action
    if not progress['games_played']:
        return None
    path = Path(root) / f'data/selfplay/{progress["run_id"]}-{progress["next_game_index"]-1}.npz'
    try:
        record = load_game(path)
    except ValueError:
        # Replay/checkpoint can still train if an optional watch archive is absent.
        return None
    return {**record.metadata(), 'moves': [decode_action(s.action).uci() for s in record.samples]}


def compact_metrics(metrics):
    return [{k: v for k, v in m.items() if k != 'inspection'} for m in metrics[-1000:]]


def snapshot(controller, telemetry=True):
    from fly_chess.neural.network import module_inventory
    p = dict(controller.progress)
    if getattr(controller, '_ui_watch_index', None) != p['next_game_index']:
        controller._ui_watch = watch_record(controller.root, p)
        controller._ui_watch_index = p['next_game_index']
    return {'kind': 'state', 'progress': p, 'latest': str(controller.latest),
        'configuration': asdict(controller.config), 'device': str(controller.device),
        'metrics': compact_metrics(controller.metrics), 'evaluations': controller.evaluations[-1000:],
        'metrics_omitted': max(0, len(controller.metrics)-1000),
        'inventory': module_inventory(controller.model),
        'inspection': controller.last_inspection if telemetry else None,
        'watch': controller._ui_watch,
        'learning_rate': controller.optimizer.param_groups[0]['lr']}


def training_entry(root, config, resume, generations, reports, commands, pause, stop, save, telemetry,
                   interval, module, load_only, gradients):
    """Child-process entry point; never imports or operates Pygame."""
    from fly_chess.training.controller import TrainingController
    try:
        if load_only:
            from fly_chess.storage.checkpoints import read_checkpoint
            from fly_chess.neural.network import PolicyValueNetwork, NetworkSpec, module_inventory
            data = read_checkpoint(Path(resume), Path(root))
            result = {'kind': 'state', 'progress': data['progress'], 'latest': str(resume),
                'configuration': data['configuration'], 'device': data['runtime']['device'],
                'metrics': compact_metrics(data['metrics']), 'evaluations': data.get('evaluations', [])[-1000:],
                'metrics_omitted': max(0, len(data['metrics'])-1000), 'inspection': None,
                'watch': watch_record(root, data['progress']),
                'inventory': module_inventory(PolicyValueNetwork(NetworkSpec(**data['spec']))),
                'learning_rate': data['optimizer']['param_groups'][0]['lr']}
            # A persisted running status describes the saved boundary, not a live process.
            result['progress'] = {**result['progress'], 'status': 'loaded'}
            reports.put(result, timeout=1)
            return
        with TrainingController(Path(root), config if resume is None else None, resume=Path(resume) if resume else None) as controller:
            controller.pause_event, controller.stop_event, controller.save_event = pause, stop, save
            controller.inspection_interval_override = interval
            controller.activation_module = module if telemetry else None
            def boundary(c):
                nonlocal telemetry, interval, module, gradients
                while True:
                    try:
                        settings = commands.get_nowait()
                    except Empty:
                        break
                    telemetry, interval, module = settings['telemetry'], settings['interval'], settings['module']
                    gradients = settings['gradients']
                c.inspection_interval_override = interval if telemetry else 0
                from fly_chess.neural.network import module_inventory
                names = {m['name'] for m in module_inventory(c.model)}
                c.activation_module = (module if module in names else 'stem') if telemetry and module else None
                c.inspect_gradients = gradients
                result = snapshot(c, telemetry)
                result['saved'] = True  # Called only after a durable controller boundary.
                offer(reports, result)
            boundary(controller)
            controller.run(generations, on_boundary=boundary)
            result = snapshot(controller, telemetry)
            result['final'] = True
            # Ensure final state follows earlier queue messages. Parent keeps draining on close.
            while True:
                try:
                    reports.put(result, timeout=.1)
                    break
                except Full:
                    try:
                        reports.get_nowait()
                    except Empty:
                        pass
    except Exception as exc:
        offer(reports, {'kind': 'error', 'error': f'{type(exc).__name__}: {exc}', 'final': True})


def human_entry(root, base, config, epochs, reports, stop):
    try:
        from fly_chess.training.human import train_human_candidate
        report = train_human_candidate(root, base, config, epochs=epochs, cancel=stop.is_set,
            on_progress=lambda report: offer(reports, {'kind': 'human', 'report': report}))
        final = {'kind': 'human', 'report': report, 'final': True}
        while True:
            try:
                reports.put(final, timeout=.1)
                break
            except Full:
                try:
                    reports.get_nowait()
                except Empty:
                    pass
    except Exception as exc:
        offer(reports, {'kind': 'error', 'error': f'{type(exc).__name__}: {exc}', 'final': True})


class TrainingWorker:
    def __init__(self):
        self.context = mp.get_context('spawn')
        self.process = None
        self.reports = self.commands = None
        self.pause = self.stop = self.save = None
        self.final_seen = False
        self.exited_at = None

    @property
    def busy(self):
        return self.process is not None and self.process.is_alive()

    def start(self, root, config, *, resume=None, generations=1, telemetry=True, interval=8, module='stem', load_only=False, gradients=True,
              human_base=None, human_epochs=1):
        if self.busy:
            return False
        self.cleanup()
        self.reports = self.context.Queue(maxsize=3)
        self.commands = self.context.Queue(maxsize=2)
        self.pause, self.stop, self.save = (self.context.Event() for _ in range(3))
        self.final_seen = False
        self.exited_at = None
        args = (str(root), config,
            str(resume) if resume else None, generations, self.reports, self.commands, self.pause,
            self.stop, self.save, telemetry, interval, module, load_only, gradients)
        target = training_entry
        if human_base is not None:
            target = human_entry
            args = (str(root), str(human_base), config, human_epochs, self.reports, self.stop)
        self.process = self.context.Process(target=target, args=args, name='fly-training')
        self.process.start()
        return True

    def configure(self, telemetry, interval, module, gradients=True):
        if self.busy:
            offer(self.commands, {'telemetry': telemetry, 'interval': interval, 'module': module, 'gradients': gradients})

    def request(self, action):
        event = getattr(self, action)
        if self.busy and event:
            event.set()

    def drain(self):
        output = []
        if self.reports:
            while True:
                try:
                    item = self.reports.get_nowait()
                    self.final_seen |= bool(item.get('final')) or item['kind'] == 'error' or item.get('progress', {}).get('status') == 'loaded'
                    output.append(item)
                except Empty:
                    break
        if self.process is not None and not self.busy and not self.final_seen:
            self.exited_at = self.exited_at or time.monotonic()
            if time.monotonic()-self.exited_at > .5:
                self.final_seen = True
                output.append({'kind': 'error', 'error': f'Training process exited without a final report (code {self.process.exitcode}). Reload latest checkpoint.'})
        return output

    def cleanup(self):
        if self.process:
            if self.busy:
                raise RuntimeError('Stop training and drain reports before closing the worker')
            self.process.join(timeout=0)
            self.process.close()
            self.process = None
        for queue in (self.reports, self.commands):
            if queue is not None:
                queue.close()
                queue.cancel_join_thread()
