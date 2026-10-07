"""Spawn-safe workers; only the parent writes archives and the replay database."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import re
from uuid import uuid4

from fly_chess.config import Config
from fly_chess.core.rules import ChessGame
from fly_chess.neural.inference import Evaluator
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import load_weights
from fly_chess.search.mcts import MCTS, SearchSettings
from fly_chess.selfplay.game import play_game
from fly_chess.storage.games import save_game
from fly_chess.training.replay import ReplayBuffer

_worker = None


def game_seed(base_seed: int, index: int) -> int:
    if type(base_seed) is not int or type(index) is not int or not 0 <= base_seed < 2**32 or not 0 <= index < 2**32:
        raise ValueError("Seed and game index must be uint32 integers")
    # Odd multiplier is a bijection modulo 2**32: distinct indices never collide.
    return (base_seed + index * 0x9E3779B9) % 2**32


def _initialize(config: Config, weights: str | None, event):
    global _worker
    seed_everything(config.seed, deterministic=config.deterministic, cpu_threads=config.cpu_threads)
    device = select_device(config.device).device
    if weights:
        model, model_id = load_weights(Path(weights), device=device)
    else:
        model = PolicyValueNetwork(NetworkSpec(config.network_channels, config.residual_blocks)).to(device)
        model_id = f"untrained-seed-{config.seed}"
    _worker = config, Evaluator(model, model_id=model_id), event


def _generate(job):
    config, evaluator, event = _worker
    game_id, index, fen = job
    seed = game_seed(config.seed, index)
    seed_everything(seed, deterministic=config.deterministic, cpu_threads=config.cpu_threads)
    search = MCTS(evaluator, SearchSettings(simulations=config.mcts_simulations, cpuct=config.mcts_cpuct,
        max_depth=config.mcts_max_depth, max_nodes=config.mcts_max_nodes, use_mcts=config.mcts_enabled), seed=seed)
    start = ChessGame.from_fen(fen, claim_draws=config.claim_draws) if fen else ChessGame(claim_draws=config.claim_draws)
    return play_game(search, game_id=game_id, seed=seed, start=start, max_plies=config.max_game_plies,
                     cancel=event.is_set)


def run_batch(config: Config, root: Path, *, weights: Path | None = None, start_fen: str | None = None,
              cancel=None, start_index: int = 0, on_game=None, batch_id: str | None = None,
              replay_path: Path | None = None) -> dict:
    """Single-owner batch; deterministic game order independent of worker scheduling.

    cancel is a parent-side callable. on_game receives a small committed progress record.
    A wave has at most worker-count results in memory; no unbounded task submission.
    """
    game_seed(config.seed, start_index)
    game_seed(config.seed, start_index + config.games_per_generation - 1)
    if start_fen:
        ChessGame.from_fen(start_fen, claim_draws=config.claim_draws)
    root = Path(root).resolve()
    archive = root / "data/selfplay"
    archive.mkdir(parents=True, exist_ok=True)
    weights_name = str(Path(weights).resolve()) if weights else None
    if weights_name and not Path(weights_name).is_file():
        raise ValueError(f"Weights file does not exist: {weights_name}")
    ctx = mp.get_context("spawn")
    event = ctx.Event()
    batch_id = batch_id or uuid4().hex
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", batch_id) is None:
        raise ValueError("batch_id must be a safe alphanumeric identifier")
    workers = min(config.selfplay_workers, config.games_per_generation)
    committed = []
    executor = None

    class SerialCancellation:
        def is_set(self):
            return event.is_set() or (cancel is not None and cancel())

    try:
        with ReplayBuffer(replay_path or root / "data/replay_buffer/replay.sqlite3", config.replay_buffer_size) as replay:
            if cancel is not None and cancel():
                return {"batch_id": batch_id, "games": [], "cancelled": True, "replay": replay.statistics()}
            if workers == 1:
                _initialize(config, weights_name, SerialCancellation())
            else:
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                    initializer=_initialize, initargs=(config, weights_name, event))
            for offset in range(0, config.games_per_generation, workers):
                if event.is_set() or (cancel is not None and cancel()):
                    event.set()
                    break
                jobs = [(f"{batch_id}-{start_index + i}", start_index + i, start_fen)
                        for i in range(offset, min(offset + workers, config.games_per_generation))]
                if executor is None:
                    records = [_generate(jobs[0])]
                else:
                    futures = [executor.submit(_generate, job) for job in jobs]
                    pending = set(futures)
                    while pending:
                        if cancel is not None and cancel():
                            event.set()
                        _, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        # Surface a failed worker promptly and stop siblings in finally.
                        for future in futures:
                            if future.done() and future.exception() is not None:
                                future.result()
                    records = [future.result() for future in futures]
                for record in records:
                    path = archive / f"{record.game_id}.npz"
                    save_game(record, path)  # Durable recovery source precedes replay transaction.
                    replay.add_game(record)
                    progress = {"game_id": record.game_id, "seed": record.seed, "status": record.status,
                        "termination": record.termination, "plies": len(record.samples), "archive": str(path)}
                    committed.append(progress)
                    if on_game is not None:
                        on_game(progress)
            return {"batch_id": batch_id, "games": committed, "cancelled": event.is_set() or any(g["status"] == "aborted" for g in committed),
                    "configuration": asdict(config), "replay": replay.statistics()}
    finally:
        event.set()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        global _worker
        _worker = None
