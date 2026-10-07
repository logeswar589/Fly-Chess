"""Versioned inference weights; the reader also accepts full training checkpoints."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch

from fly_chess.core.actions import ACTION_SCHEMA_VERSION
from fly_chess.core.encoding import ENCODING_SCHEMA_VERSION
from fly_chess.neural.network import NETWORK_SCHEMA_VERSION, NetworkSpec, PolicyValueNetwork

SCHEMAS = {"format_version": 1, "network_schema": NETWORK_SCHEMA_VERSION,
           "action_schema": ACTION_SCHEMA_VERSION, "encoding_schema": ENCODING_SCHEMA_VERSION}


def save_weights(model: PolicyValueNetwork, path: Path, *, model_id: str = "untrained") -> None:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Cannot save non-finite weights")
    payload = {**SCHEMAS, "spec": asdict(model.spec), "model_id": model_id, "state_dict": state}
    path = Path(path)
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


def load_weights(path: Path, *, device: torch.device | str = "cpu") -> tuple[PolicyValueNetwork, str]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in SCHEMAS.items()):
            raise ValueError("Incompatible weight schema")
        if not isinstance(payload.get("model_id"), str) or not payload["model_id"]:
            raise ValueError("Missing model identity")
        spec = NetworkSpec(**payload["spec"])
        state = payload["state_dict"]
        if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()
        ):
            raise ValueError("Invalid or non-finite weights")
        # Loading an inference model must not consume the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = PolicyValueNetwork(spec)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"Cannot load weights from {path}: {exc}") from exc
    return model.to(device).eval(), payload["model_id"]
