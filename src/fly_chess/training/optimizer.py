"""Real policy/value optimization, with optional detached learning diagnostics."""

import numpy as np
import torch


def train_step(model, optimizer, samples, *, regularization: float, gradient_clip: float, inspect=False,
               activation_module=None, inspect_gradients=True):
    if not samples:
        raise ValueError("Cannot optimize an empty batch")
    model.train()
    device = next(model.parameters()).device
    states = torch.from_numpy(np.stack([s.state for s in samples])).to(device)
    targets = torch.tensor([s.target for s in samples], dtype=torch.float32, device=device)
    if not torch.isfinite(targets).all() or (targets.abs() > 1).any():
        raise ValueError("Training requires finite outcome targets in [-1, 1]")
    optimizer.zero_grad(set_to_none=True)
    captures = {}
    def capture(label):
        from fly_chess.neural.inspection import ActivationCapture, InspectionRequest
        request = InspectionRequest('training-'+label, max_values=65536, values_per_tensor=65536,
                                    selected_module=activation_module)
        with torch.no_grad(), ActivationCapture(model, request, {'phase': label}) as collector:
            model(states[:1])
        captures[label] = collector.snapshot
        selected = dict(model.named_modules()).get('' if activation_module == '<network>' else activation_module)
        captures[label]['weights'] = [{
            'name': name, 'shape': list(parameter.shape), 'total_elements': parameter.numel(),
            'sample_values': parameter.detach().reshape(-1)[:4096].float().cpu().tolist()}
            for name, parameter in selected.named_parameters(recurse=False)] if selected is not None else []
    if inspect and activation_module is not None:
        capture('before')
    logits, values = model(states)
    policy_terms = []
    for i, sample in enumerate(samples):
        ids = torch.as_tensor(sample.legal_actions.astype(np.int64), device=device)
        target = torch.as_tensor(sample.policy, dtype=torch.float32, device=device)
        policy_terms.append(-(target * torch.log_softmax(logits[i, ids], dim=0)).sum())
    policy_loss = torch.stack(policy_terms).mean()
    value_loss = torch.nn.functional.mse_loss(values, targets)
    reg_loss = regularization * sum(p.square().sum() for p in model.parameters())
    loss = policy_loss + value_loss + reg_loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss; update not applied")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()} if inspect and inspect_gradients else {}
    layers = {}
    if inspect and inspect_gradients:
        for name, module in model.named_modules():
            parameters = list(module.parameters(recurse=False))
            if parameters:
                layers[name] = {"gradient_norm_after_clip": float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)).item()),
                                "weight_norm_before": float(torch.sqrt(sum(p.square().sum() for p in parameters)).item())}
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite updated weights; reload the last durable checkpoint")
    report = {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
              "regularization_loss": float(reg_loss.detach()), "loss": float(loss.detach()),
              "gradient_norm_before_clip": float(grad_norm), "batch_size": len(samples)}
    if inspect:
        with torch.no_grad():
            after_logits, after_values = model(states[:1])
            ids = torch.as_tensor(samples[0].legal_actions.astype(np.int64), device=device)
            for name, module in model.named_modules():
                if name in layers:
                    delta = sum((p - before[f"{name}.{key}" if name else key]).square().sum()
                                for key, p in module.named_parameters(recurse=False))
                    layers[name]["update_norm"] = float(torch.sqrt(delta))
            report["inspection"] = {"layers": layers, "sample_index": 0,
                "legal_actions": samples[0].legal_actions.tolist(), "target_policy": samples[0].policy.tolist(),
                "policy_before": torch.softmax(logits[0, ids].detach(), 0).cpu().tolist(),
                "policy_after": torch.softmax(after_logits[0, ids], 0).cpu().tolist(),
                "value_before": float(values[0].detach()), "value_after": float(after_values[0]),
                "target_value": float(targets[0]), "perspective": "player_to_move"}
        if activation_module is not None:
            capture('after')
            report['activation_capture'] = {'module': activation_module, **captures}
    return report
