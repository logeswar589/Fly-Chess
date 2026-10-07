"""Compressed, versioned, non-pickle game archives with atomic replacement."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from fly_chess.core.encoding import INPUT_PLANES
from fly_chess.selfplay.records import GameRecord, Sample, SCHEMAS


def save_game(record: GameRecord, path: Path) -> None:
    record.validate()
    count = len(record.samples)
    metadata = {**record.metadata(), "samples": [{"action": s.action, "white_to_move": s.white_to_move,
        "value_prediction": s.value_prediction, "policy_source": s.policy_source, "target": s.target} for s in record.samples]}
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, allow_nan=False)),
        "states": np.stack([s.state for s in record.samples]) if count else np.empty((0, INPUT_PLANES, 8, 8), dtype=np.float32),
        "offsets": np.asarray([0, *np.cumsum([len(s.legal_actions) for s in record.samples]).tolist()], dtype=np.int64),
        "legal": np.concatenate([s.legal_actions for s in record.samples]).astype(np.uint16) if count else np.empty(0, dtype=np.uint16),
        "policy": np.concatenate([s.policy for s in record.samples]).astype(np.float32) if count else np.empty(0, dtype=np.float32),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_game(path: Path) -> GameRecord:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if any(meta.pop(key, None) != value for key, value in SCHEMAS.items()):
                raise ValueError("Incompatible game schema")
            sample_meta = meta.pop("samples")
            states, offsets, legal, policy = (data[name] for name in ("states", "offsets", "legal", "policy"))
            count = len(sample_meta)
            if (states.shape != (count, INPUT_PLANES, 8, 8) or offsets.shape != (count + 1,)
                    or offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != len(legal)
                    or np.any(np.diff(offsets) < 0) or legal.ndim != 1 or policy.shape != legal.shape):
                raise ValueError("Malformed game arrays")
            record = GameRecord(**meta)
            for i, fields in enumerate(sample_meta):
                start, end = int(offsets[i]), int(offsets[i + 1])
                record.samples.append(Sample(states[i].copy(), legal_actions=legal[start:end].copy(),
                    policy=policy[start:end].copy(), **fields))
        record.validate()
        return record
    except Exception as exc:
        raise ValueError(f"Cannot load self-play game {path}: {exc}") from exc
