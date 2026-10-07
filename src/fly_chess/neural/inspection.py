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
            "kind": "sampled", "tensors": {}, "brain": {}, "connections": [],
        }
        self.remaining = request.max_values
        self.handles = []
        self.brain_remaining = 8192 if request.brain else 0
        self.leaf_names = {name for name, module in model.named_modules() if not list(module.children())}

    def _record(self, name, output, module=None, inputs=()):
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
                if index == 0:
                    self._connections(name, module, inputs, tensor, indices[:12])
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

    def _connections(self, name, module, inputs, output, targets):
        """Sample real weighted terms, not proximity edges in the illustrative layout.

        One strongest absolute weight per sampled output; convolution samples use
        their actual receptive-field coordinate. Bias, other terms and nonlinear
        operations are deliberately not represented by these bounded edges.
        """
        if not inputs or not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            return
        source = inputs[0].detach()
        weights = module.weight.detach()
        if source.shape[0] != 1 or not len(targets) or self.brain_remaining < len(targets):
            return
        if isinstance(module, torch.nn.Linear):
            if source.ndim != 2:
                return
            rows = weights[targets]
            source_ids = rows.abs().argmax(dim=1)
            selected_weights = rows.gather(1, source_ids[:, None]).squeeze(1)
        else:
            if module.groups != 1 or isinstance(module.padding, str):
                return
            _, channels, height, width = source.shape
            oh, ow = output.shape[-2:]
            out_channels = targets // (oh*ow)
            oy, ox = (targets % (oh*ow)) // ow, targets % ow
            rows = weights[out_channels].reshape(len(targets), -1)
            offsets = rows.abs().argmax(dim=1)
            kh, kw = module.kernel_size
            ic = offsets // (kh*kw)
            iy = oy*module.stride[0]-module.padding[0]+((offsets % (kh*kw))//kw)*module.dilation[0]
            ix = ox*module.stride[1]-module.padding[1]+(offsets % kw)*module.dilation[1]
            valid = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
            source_ids = ((ic*height+iy)*width+ix)[valid]
            selected_weights = rows.gather(1, offsets[:, None]).squeeze(1)[valid]
            targets = targets[valid]
        values = source.reshape(-1)[source_ids].float()
        ids, dst = source_ids.cpu().tolist(), targets.cpu().tolist()
        values, weights = values.cpu().tolist(), selected_weights.float().cpu().tolist()
        source_key = name+'/input:0'
        self.snapshot['brain'][source_key] = {'indices': ids, 'values': values,
            'shape': list(source.shape), 'total': source.numel()}
        self.brain_remaining -= len(ids)
        for src, target, value, weight in zip(ids, dst, values, weights):
            self.snapshot['connections'].append({'source': source_key, 'source_index': src,
                'target': name+':0', 'target_index': target, 'weight': weight,
                'input': value, 'contribution': value*weight})

    def __enter__(self):
        inventory = dict(self.model.named_modules())
        names = {name or "<network>" for name in inventory}
        if self.request.selected_module is not None and self.request.selected_module not in names:
            raise ValueError(f"Unknown module: {self.request.selected_module}")
        for name, module in inventory.items():
            label = name or "<network>"
            self.handles.append(module.register_forward_hook(
                lambda module, inputs, output, label=label: self._record(label, output, module, inputs)
            ))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
