"""Single-owner generation state machine with game/minibatch durability."""

from dataclasses import asdict, replace
from pathlib import Path
import shutil
import threading
from time import monotonic
from uuid import uuid4

from filelock import FileLock, Timeout
import numpy as np
import torch

from fly_chess.config import Config
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import SCHEMAS, save_weights, load_weights
from fly_chess.evaluation.arena import ArenaSettings, evaluate_candidate
from fly_chess.selfplay.workers import game_seed, run_batch
from fly_chess.storage.checkpoints import (atomic_torch_save, capture_rng, checked_replay_path,
    read_checkpoint, restore_rng, snapshot_replay)
from fly_chess.storage.games import load_game
from fly_chess.storage.workspace import Workspace
from fly_chess.training.optimizer import train_step
from fly_chess.training.replay import ReplayBuffer


class TrainingController:
    def __init__(self, root: Path, config: Config | None = None, *, resume: Path | None = None):
        self.root = Path(root).resolve()
        Workspace(self.root).initialize()
        self.lock = FileLock(str(self.root / "models/.training.lock"))
        try:
            self.lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("Another training controller owns this workspace") from exc
        self.latest = self.root / "models/fly_latest.pt"
        self.replay_path = self.root / "data/replay_buffer/training.sqlite3"
        self.pause_event, self.stop_event, self.save_event = (threading.Event() for _ in range(3))
        self._running = False
        self._last_tick = None
        self._manifest = None
        self._replay_dirty = True
        self.inspection_interval_override = None
        self.activation_module = None
        self.last_inspection = None
        self.inspect_gradients = True
        try:
            data = read_checkpoint(Path(resume), self.root) if resume is not None else None
            if data:
                saved_config = Config(**data["configuration"])
                if config is not None and config != saved_config:
                    raise ValueError("Resume configuration must match the saved checkpoint; omit --config")
                self.config = saved_config
            else:
                if self.latest.exists() or self.replay_path.exists():
                    raise ValueError("Training state already exists; resume it or choose a new workspace")
                self.config = config or Config()
            cfg = self.config
            seed_everything(cfg.seed, deterministic=cfg.deterministic, cpu_threads=cfg.cpu_threads)
            self.device = select_device(cfg.device).device
            self.model = PolicyValueNetwork(NetworkSpec(cfg.network_channels, cfg.residual_blocks)).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.learning_rate_decay)
            self.rng = np.random.default_rng(cfg.seed)
            if data:
                self.model.load_state_dict(data["state_dict"], strict=True)
                self.optimizer.load_state_dict(data["optimizer"])
                self.scheduler.load_state_dict(data["scheduler"])
                self.progress, self.metrics = data["progress"], data["metrics"]
                self.evaluations = data.get("evaluations", [])
                self._validate_progress()
                source = checked_replay_path(self.root, data["replay"])
                temporary = self.replay_path.with_name(f"restore-{uuid4().hex}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    temporary.replace(self.replay_path)
                finally:
                    temporary.unlink(missing_ok=True)
                self._manifest = data["replay"]
                self._replay_dirty = False
                restore_rng(data["rng"], self.rng)
            else:
                self.progress = {"run_id": uuid4().hex, "generation": 1, "completed_generations": 0,
                    "stage": "selfplay", "games_in_generation": 0, "updates_in_generation": 0,
                    "next_game_index": 0, "training_steps": 0, "games_played": 0,
                    "completed_games": 0, "truncated_games": 0, "white_wins": 0, "black_wins": 0,
                    "draws": 0, "total_plies": 0, "active_seconds": 0.0, "status": "ready",
                    "estimated_elo": None, "evaluation": {"status": "not_evaluated"}}
                self.metrics = []
                self.evaluations = []
                with ReplayBuffer(self.replay_path, cfg.replay_buffer_size):
                    pass
                self.save()
        except BaseException as exc:
            self.lock.release()
            if resume is not None and isinstance(exc, Exception) and not isinstance(exc, ValueError):
                raise ValueError(f"Cannot restore training state: {exc}") from exc
            raise

    def _validate_progress(self):
        p, cfg = self.progress, self.config
        if p["stage"] not in ("selfplay", "optimize", "evaluate", "finish") or not 0 <= p["games_in_generation"] <= cfg.games_per_generation or not 0 <= p["updates_in_generation"] <= cfg.updates_per_generation:
            raise ValueError("Invalid checkpoint stage/counters")
        for key in ("generation", "completed_generations", "training_steps", "next_game_index", "games_played"):
            if type(p[key]) is not int or p[key] < 0:
                raise ValueError("Invalid checkpoint progress")
        if p["generation"] != p["completed_generations"] + 1 or p["next_game_index"] != p["games_played"]:
            raise ValueError("Inconsistent checkpoint progress")
        if p["stage"] != "selfplay" and p["games_in_generation"] != cfg.games_per_generation:
            raise ValueError("Optimization checkpoint has incomplete self-play")
        if p["stage"] in ("evaluate", "finish") and p["updates_in_generation"] != cfg.updates_per_generation:
            raise ValueError("Evaluation checkpoint has incomplete optimization")
        if len(p["run_id"]) != 32 or any(c not in "0123456789abcdef" for c in p["run_id"]):
            raise ValueError("Invalid checkpoint run identity")

    @property
    def model_id(self):
        return f"{self.progress['run_id']}:step-{self.progress['training_steps']}"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._running:
            raise RuntimeError("Request stop and join the training worker before closing")
        self.lock.release()

    def request_pause(self):
        self.pause_event.set()

    def request_stop(self):
        self.stop_event.set()

    def request_save(self):
        self.save_event.set()

    def _tick(self):
        if self._last_tick is not None:
            now = monotonic()
            self.progress["active_seconds"] += now - self._last_tick
            self._last_tick = now

    def save(self, *, generation_file: int | None = None):
        """Owning worker only; other threads use request_save()."""
        self._tick()
        if self._replay_dirty:
            self._manifest = snapshot_replay(self.root, self.replay_path)
            self._replay_dirty = False
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Refusing to checkpoint non-finite weights")
        payload = {**SCHEMAS, "training_schema": 1, "spec": asdict(self.model.spec),
            "model_id": self.model_id, "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "rng": capture_rng(self.rng), "configuration": asdict(self.config),
            "progress": self.progress, "metrics": self.metrics, "replay": self._manifest,
            "evaluations": self.evaluations,
            "runtime": {"torch": str(torch.__version__), "device": str(self.device)}}
        if generation_file is not None:
            atomic_torch_save(payload, self.root / f"models/fly_generation_{generation_file:03d}.pt")
        atomic_torch_save(payload, self.latest)

    def _commit_game(self, archive: Path):
        record = load_game(archive)
        index = self.progress["next_game_index"]
        expected_id = f"{self.progress['run_id']}-{index}"
        if record.game_id != expected_id or record.seed != game_seed(self.config.seed, index) or record.model_id != self.model_id or record.status == "aborted":
            raise ValueError("Recovery archive does not match the expected game/model/seed")
        with ReplayBuffer(self.replay_path, self.config.replay_buffer_size) as replay:
            replay.add_game(record)
        p = self.progress
        p["games_in_generation"] += 1
        p["games_played"] += 1
        p["next_game_index"] += 1
        p["total_plies"] += len(record.samples)
        p["average_game_length"] = p["total_plies"] / p["games_played"]
        p["completed_games" if record.status == "completed" else "truncated_games"] += 1
        if record.status == "completed":
            p[{1: "white_wins", -1: "black_wins", 0: "draws"}[record.white_result]] += 1
        self._replay_dirty = True
        self.save()

    def _selfplay_wave(self):
        p, cfg = self.progress, self.config
        # Recover a game durably archived just before a crash but not checkpointed.
        while p["games_in_generation"] < cfg.games_per_generation:
            archive = self.root / f"data/selfplay/{p['run_id']}-{p['next_game_index']}.npz"
            if not archive.exists():
                break
            self._commit_game(archive)
        remaining = cfg.games_per_generation - p["games_in_generation"]
        if remaining:
            weights = self.root / f"models/selfplay/{p['run_id']}-{p['training_steps']}.pt"
            save_weights(self.model, weights, model_id=self.model_id)
            wave = replace(cfg, games_per_generation=min(cfg.selfplay_workers, remaining))
            parent_rng = capture_rng(self.rng)
            def commit(info):
                restore_rng(parent_rng, self.rng)
                self._commit_game(Path(info["archive"]))
            try:
                run_batch(wave, self.root, weights=weights, start_index=p["next_game_index"],
                    batch_id=p["run_id"], replay_path=self.replay_path, on_game=commit)
            finally:
                restore_rng(parent_rng, self.rng)
        if p["games_in_generation"] == cfg.games_per_generation:
            p["stage"] = "optimize"
            self.save()

    def _update(self):
        cfg, p = self.config, self.progress
        with ReplayBuffer(self.replay_path, cfg.replay_buffer_size) as replay:
            if not len(replay):
                raise ValueError("No replay positions are available for training")
            rows = replay.sample(min(cfg.batch_size, len(replay)), self.rng)
        learning_rate = self.optimizer.param_groups[0]["lr"]
        interval = cfg.training_inspection_interval if self.inspection_interval_override is None else self.inspection_interval_override
        inspect = bool(interval and (p["training_steps"] + 1) % interval == 0)
        report = train_step(self.model, self.optimizer, [r["sample"] for r in rows],
            regularization=cfg.weight_decay, gradient_clip=cfg.gradient_clip, inspect=inspect,
            activation_module=self.activation_module if inspect else None, inspect_gradients=self.inspect_gradients)
        activation_capture = report.pop('activation_capture', None)
        self.scheduler.step()
        p["training_steps"] += 1
        p["updates_in_generation"] += 1
        self.metrics.append({**report, "generation": p["generation"], "step": p["training_steps"],
            "learning_rate": learning_rate, "games_played": p['games_played'],
            "average_game_length": p.get('average_game_length'),
            "sample_keys": [[r["game_id"], r["ply"]] for r in rows]})
        if inspect:
            from fly_chess.core.rules import ChessGame
            source_fen = None
            try:
                record = load_game(self.root / f'data/selfplay/{rows[0]["game_id"]}.npz')
                source = ChessGame.from_fen(record.root_fen)
                for move in record.prefix_moves:
                    source.push(move)
                for sample in record.samples[:rows[0]['ply']]:
                    source.push_action(sample.action)
                source_fen = source.board.fen()
            except ValueError:
                pass  # Missing optional archive must not invalidate a valid optimizer update.
            self.last_inspection = {**(activation_capture or {}), 'fen': source_fen,
                'model_id': self.model_id, 'step': p['training_steps'],
                'sample_key': [rows[0]['game_id'], rows[0]['ply']], 'learning': report['inspection']}
        self.save()  # The optimizer state and progress marker commit together.

    def _evaluate(self):
        cfg, p = self.config, self.progress
        if not cfg.evaluation_enabled:
            p["evaluation"] = {"status": "disabled"}
            p["stage"] = "finish"
            self.save()
            return
        candidate = self.root / f"models/candidates/{p['run_id']}-{p['generation']}.pt"
        if not candidate.exists():
            # Freeze the full durable state, so a promoted best can also resume training.
            payload = torch.load(self.latest, map_location="cpu", weights_only=True)
            atomic_torch_save(payload, candidate)
        else:
            existing, identity = load_weights(candidate)
            if identity != self.model_id or any(not torch.equal(value.detach().cpu(), existing.state_dict()[name])
                                               for name, value in self.model.state_dict().items()):
                raise ValueError("Existing candidate snapshot does not match this generation")
        parent_rng = capture_rng(self.rng)
        try:
            report = evaluate_candidate(self.root, candidate,
                evaluation_id=f"{p['run_id']}-generation-{p['generation']}", device=str(self.device),
                settings=ArenaSettings(pairs=cfg.evaluation_pairs, min_pairs=cfg.evaluation_min_pairs,
                    simulations=cfg.evaluation_simulations, max_plies=cfg.evaluation_max_plies,
                    seed=(cfg.seed + p["generation"]) % 2**32),
                cancel=lambda: self.pause_event.is_set() or self.stop_event.is_set())
        finally:
            seed_everything(cfg.seed, deterministic=cfg.deterministic, cpu_threads=cfg.cpu_threads)
            restore_rng(parent_rng, self.rng)
        compact = {key: report.get(key) for key in ("evaluation_id", "status", "candidate_id", "incumbent_id",
                   "settings", "promoted", "reason", "statistics")}
        p["evaluation"] = compact
        p["estimated_elo"] = None  # No calibrated absolute Elo; relative estimates live in evaluation statistics.
        if report["status"] in ("complete", "baseline"):
            self.evaluations.append({"generation": p["generation"], "games_played": p["games_played"], **compact})
            p["stage"] = "finish"
        self.save()

    def run(self, generations: int = 1, *, on_boundary=None) -> dict:
        if not self.lock.is_locked or self.progress["status"] == "failed":
            raise RuntimeError("Controller is closed or failed; reload the last durable checkpoint")
        if self._running or type(generations) is not int or generations < 1:
            raise ValueError("Run requires a positive generation count and an idle controller")
        self._running = True
        self._last_tick = monotonic()
        p = self.progress
        target = p["completed_generations"] + generations
        p["status"] = "running"
        try:
            while True:
                if self.stop_event.is_set() or self.pause_event.is_set():
                    p["status"] = "stopped" if self.stop_event.is_set() else "paused"
                    self.stop_event.clear()
                    self.pause_event.clear()
                    self.save()
                    break
                if self.save_event.is_set():
                    self.save()
                    self.save_event.clear()
                    if on_boundary is not None:
                        on_boundary(self)
                if p["completed_generations"] >= target:
                    p["status"] = "ready"
                    self.save()
                    break
                if p["stage"] == "selfplay":
                    self._selfplay_wave()
                elif p["stage"] == "optimize" and p["updates_in_generation"] < self.config.updates_per_generation:
                    self._update()
                elif p["stage"] == "optimize":
                    p["stage"] = "evaluate"
                    self.save()
                elif p["stage"] == "evaluate":
                    self._evaluate()
                else:
                    finished = p["generation"]
                    p.update(completed_generations=finished, generation=finished + 1, stage="selfplay",
                             games_in_generation=0, updates_in_generation=0)
                    self.save(generation_file=finished if finished % self.config.checkpoint_interval == 0 else None)
                if on_boundary is not None:
                    on_boundary(self)
        except BaseException:
            p["status"] = "failed"
            # Never save a partially applied update after an exception. Disk retains last good boundary.
            raise
        finally:
            self._tick()
            self._last_tick = None
            self._running = False
        return {"progress": dict(p), "latest": str(self.latest), "device": str(self.device),
                "last_update": self.metrics[-1] if self.metrics else None}
