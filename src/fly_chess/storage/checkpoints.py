"""Atomic training state plus immutable, checksum-verified replay snapshots."""

import hashlib
from contextlib import closing
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from uuid import uuid4

import numpy as np
import torch

from fly_chess.neural.weights import SCHEMAS


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_replay(root: Path, replay_path: Path) -> dict:
    destination = root / "data/replay_buffer/snapshots" / f"{uuid4().hex}.sqlite3"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(replay_path)) as source, closing(sqlite3.connect(destination)) as target:
        source.backup(target)
    # Commit+close above, then flush before a checkpoint can reference the snapshot.
    with destination.open("rb+") as stream:
        os.fsync(stream.fileno())
    return {"path": destination.relative_to(root).as_posix(), "sha256": file_hash(destination)}


def checked_replay_path(root: Path, manifest: dict) -> Path:
    path = (root / manifest["path"]).resolve()
    if not path.is_relative_to((root / "data/replay_buffer/snapshots").resolve()):
        raise ValueError("Replay manifest points outside the snapshot directory")
    if not path.is_file() or file_hash(path) != manifest["sha256"]:
        raise ValueError("Checkpoint replay snapshot is missing or corrupted")
    return path


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], torch.tensor(state[1].astype(np.int64)), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": generator.bit_generator.state}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG resume unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.bit_generator.state = state["sampler"]


def read_checkpoint(path: Path, root: Path) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("training_schema") != 1 or any(data.get(k) != v for k, v in SCHEMAS.items()):
            raise ValueError("Incompatible training checkpoint schema (inference-only files cannot resume)")
        required = {"state_dict", "spec", "optimizer", "scheduler", "rng", "configuration", "progress", "metrics", "replay", "model_id"}
        if not required <= data.keys():
            raise ValueError("Incomplete training checkpoint")
        if any(not isinstance(data[key], dict) for key in ("configuration", "progress", "optimizer", "scheduler", "rng", "replay")) or not isinstance(data["metrics"], list):
            raise ValueError("Malformed training checkpoint")
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        if not finite(data):
            raise ValueError("Non-finite checkpoint state")
        if not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in data["state_dict"].values()):
            raise ValueError("Invalid checkpoint weights")
        checked_replay_path(root, data["replay"])
        return data
    except Exception as exc:
        raise ValueError(f"Cannot resume checkpoint {path}: {exc}") from exc
