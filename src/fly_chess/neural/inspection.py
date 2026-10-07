"""One-forward activation samples; no graph retention, global hooks, or queues."""

from dataclasses import dataclass
from datetime import datetime, timezone

import torch


@dataclass(frozen=True)
class InspectionRequest:
    request_id: str
    max_values: int = 4096
    values_per_tensor: int = 64
    selected_module: str | None = None
    brain: bool = False

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("Inspection requires a nonempty request_id")
        for name in ("max_values", "values_per_tensor"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65536:
                raise ValueError(f"{name} must be an integer in [1, 65536]")


class ActivationCapture:
    """Use only around an isolated forward in the model's owning worker."""
    def __init__(self, model, request: InspectionRequest, metadata: dict):
        self.model = model
        self.request = request
        self.snapshot = {
            **metadata, "request_id": request.request_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "sampled", "tensors": {}, "brain": {},
        }
        self.remaining = request.max_values
        self.handles = []
        self.brain_remaining = 8192 if request.brain else 0
        self.leaf_names = {name for name, module in model.named_modules() if not list(module.children())}

    def _record(self, name, output):
        outputs = output if isinstance(output, tuple) else (output,)
        for index, tensor in enumerate(outputs):
            if not isinstance(tensor, torch.Tensor):
                continue
            key = f"{name}:{index}"
            if name in self.leaf_names and self.brain_remaining:
                flat = tensor.detach().reshape(-1)
                n = min(48, flat.numel(), self.brain_remaining)
                indices = torch.linspace(0, flat.numel()-1, n, device=flat.device).long()
                self.snapshot['brain'][key] = {
                    'indices': indices.cpu().tolist(), 'values': flat[indices].float().cpu().tolist(),
                    'shape': list(tensor.shape), 'total': flat.numel(),
                }
                self.brain_remaining -= n
            eligible = self.request.selected_module in (None, name)
            count = min(self.remaining, self.request.values_per_tensor, tensor.numel()) if eligible else 0
            # Slice on the device before transfer; materialize independent Python values.
            values = tensor.detach().reshape(-1)[:count].float().cpu().tolist() if count else []
            self.remaining -= count
            self.snapshot["tensors"][key] = {
                "shape": list(tensor.shape), "sample_offset": 0,
                "sample_values": values, "sampled_elements": count,
                "total_elements": tensor.numel(),
                "truncated": count < tensor.numel(),
            }

    def __enter__(self):
        inventory = dict(self.model.named_modules())
        names = {name or "<network>" for name in inventory}
        if self.request.selected_module is not None and self.request.selected_module not in names:
            raise ValueError(f"Unknown module: {self.request.selected_module}")
        for name, module in inventory.items():
            label = name or "<network>"
            self.handles.append(module.register_forward_hook(
                lambda module, inputs, output, label=label: self._record(label, output)
            ))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
