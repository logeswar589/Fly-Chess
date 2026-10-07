from dataclasses import dataclass

import torch
from torch import nn

from fly_chess.core.actions import ACTION_SIZE
from fly_chess.core.encoding import INPUT_PLANES

NETWORK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NetworkSpec:
    channels: int = 32
    residual_blocks: int = 2

    def __post_init__(self):
        for name, maximum in (("channels", 256), ("residual_blocks", 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = self.relu1(self.norm1(self.conv1(x)))
        return self.relu2(x + self.norm2(self.conv2(residual)))


class PolicyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Each source square produces 64 destinations * 5 promotion slots.
        self.projection = nn.Conv2d(channels, 64 * 5, 1)

    def forward(self, x):
        logits = self.projection(x)
        return logits.permute(0, 2, 3, 1).reshape(x.shape[0], ACTION_SIZE)


class PolicyValueNetwork(nn.Module):
    def __init__(self, spec: NetworkSpec | None = None):
        super().__init__()
        self.spec = spec or NetworkSpec()
        channels = self.spec.channels
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels), nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(self.spec.residual_blocks)))
        self.policy = PolicyHead(channels)
        self.value = nn.Sequential(
            nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 4 or tuple(positions.shape[1:]) != (INPUT_PLANES, 8, 8):
            raise ValueError(f"Expected [batch, {INPUT_PLANES}, 8, 8] input")
        if positions.shape[0] == 0 or positions.dtype != torch.float32:
            raise ValueError("Network input must be a nonempty float32 batch")
        shared = self.trunk(self.stem(positions))
        return self.policy(shared), self.value(shared).squeeze(-1)


def module_inventory(model: nn.Module) -> list[dict]:
    """Every module is reachable; own parameter counts avoid double counting."""
    return [
        {"name": name or "<network>", "type": type(module).__name__,
         "parameters": sum(p.numel() for p in module.parameters(recurse=False))}
        for name, module in model.named_modules()
    ]
