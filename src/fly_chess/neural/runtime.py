from dataclasses import dataclass
import os
import random

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    reason: str


def select_device(requested: str = "auto") -> DeviceSelection:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return DeviceSelection(torch.device("cpu"), "CPU explicitly selected")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch installation or on this machine")
        # Probe execution, not just device enumeration, before promising CUDA.
        probe = torch.ones((2, 2), device="cuda")
        (probe @ probe).sum().item()
        torch.cuda.synchronize()
        return DeviceSelection(torch.device("cuda"), torch.cuda.get_device_name(0))
    except (RuntimeError, AssertionError) as exc:
        if requested == "cuda":
            raise RuntimeError(f"Requested CUDA is not usable: {exc}") from exc
        return DeviceSelection(torch.device("cpu"), f"Using CPU: {exc}")


def seed_everything(seed: int, *, deterministic: bool = True, cpu_threads: int = 2) -> None:
    """Call once in each owning process, before creating networks or CUDA contexts."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(deterministic) is not bool or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("deterministic must be bool and cpu_threads a positive integer")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
