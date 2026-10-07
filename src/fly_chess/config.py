"""Strict, portable settings. Paths are supplied separately at launch."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib


class ConfigurationError(ValueError):
    """A setting cannot be used safely or is not understood."""


@dataclass(frozen=True)
class Config:
    device: str = "auto"
    batch_size: int = 32
    learning_rate: float = 0.001
    mcts_simulations: int = 32
    selfplay_workers: int = 1
    games_per_generation: int = 8
    checkpoint_interval: int = 1
    replay_buffer_size: int = 10000
    max_game_plies: int = 512
    seed: int = 42
    claim_draws: bool = True
    network_channels: int = 32
    residual_blocks: int = 2
    deterministic: bool = True
    cpu_threads: int = 2
    mcts_cpuct: float = 1.5
    mcts_max_depth: int = 128
    mcts_max_nodes: int = 10000
    mcts_enabled: bool = True
    updates_per_generation: int = 32
    weight_decay: float = 0.0001
    learning_rate_decay: float = 0.999
    gradient_clip: float = 1.0
    training_inspection_interval: int = 0
    evaluation_enabled: bool = True
    evaluation_pairs: int = 50
    evaluation_min_pairs: int = 20
    evaluation_simulations: int = 32
    evaluation_max_plies: int = 512

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigurationError("device must be auto, cpu, or cuda")
        positive = (
            "batch_size", "mcts_simulations", "selfplay_workers",
            "games_per_generation", "checkpoint_interval",
            "replay_buffer_size", "max_game_plies",
            "network_channels", "residual_blocks", "cpu_threads",
            "mcts_max_depth", "mcts_max_nodes",
            "updates_per_generation",
            "evaluation_pairs", "evaluation_min_pairs", "evaluation_simulations", "evaluation_max_plies",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ConfigurationError("seed must be an integer in [0, 2**32)")
        if type(self.claim_draws) is not bool:
            raise ConfigurationError("claim_draws must be true or false")
        if type(self.deterministic) is not bool:
            raise ConfigurationError("deterministic must be true or false")
        if type(self.mcts_enabled) is not bool:
            raise ConfigurationError("mcts_enabled must be true or false")
        if type(self.evaluation_enabled) is not bool:
            raise ConfigurationError("evaluation_enabled must be true or false")
        if type(self.mcts_cpuct) not in (int, float) or not math.isfinite(self.mcts_cpuct) or self.mcts_cpuct <= 0:
            raise ConfigurationError("mcts_cpuct must be finite and positive")
        if self.network_channels > 256 or self.residual_blocks > 16:
            raise ConfigurationError("network_channels must be <= 256 and residual_blocks <= 16")
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= 1):
            raise ConfigurationError("learning_rate must be finite and in (0, 1]")
        if self.batch_size > self.replay_buffer_size:
            raise ConfigurationError("batch_size must not exceed replay_buffer_size")
        for name, low, high in (("weight_decay", 0, 1), ("learning_rate_decay", 0, 1), ("gradient_clip", 0, 1000)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or (name != "weight_decay" and value == 0):
                raise ConfigurationError(f"Invalid {name}")
        if type(self.training_inspection_interval) is not int or self.training_inspection_interval < 0:
            raise ConfigurationError("training_inspection_interval must be a nonnegative integer")


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config()
    try:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Cannot read config {path}: {exc}") from exc
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ConfigurationError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return Config(**values)
