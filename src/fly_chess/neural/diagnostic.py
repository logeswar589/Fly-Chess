from pathlib import Path

import torch

from fly_chess.core.actions import decode_action
from fly_chess.neural.inference import Evaluator
from fly_chess.neural.inspection import InspectionRequest
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork, module_inventory
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import load_weights, save_weights


def diagnose_network(config, game, *, weights: Path | None = None,
                     save: Path | None = None, inspect: bool = False) -> dict:
    seed_everything(config.seed, deterministic=config.deterministic, cpu_threads=config.cpu_threads)
    selection = select_device(config.device)
    if weights:
        model, model_id = load_weights(weights, device=selection.device)
    else:
        model = PolicyValueNetwork(NetworkSpec(config.network_channels, config.residual_blocks)).to(selection.device)
        model_id = f"untrained-seed-{config.seed}"
    evaluator = Evaluator(model, model_id=model_id)
    result = evaluator.evaluate([game], inspection=InspectionRequest("cli-diagnostic") if inspect else None)[0]
    actions = game.actions()
    ranked = sorted(actions, key=lambda action: -float(result.policy[action]))[:10]
    if save:
        save_weights(model, save, model_id=model_id)
    return {
        "mode": "policy-only; no MCTS", "model_id": model_id,
        "weights_source": str(weights) if weights else "random initialization; not trained",
        "torch_version": torch.__version__, "device": str(selection.device),
        "device_reason": selection.reason, "parameters": sum(p.numel() for p in model.parameters()),
        "terminal": result.terminal, "value": result.value,
        "value_perspective": "player_to_move", "policy_sum": float(result.policy.sum()),
        "top_moves": [{"move": decode_action(a).uci(), "probability": float(result.policy[a])} for a in ranked],
        "position_id": result.position_id,
        "modules": module_inventory(model) if inspect else None,
        "inspection": result.inspection, "saved_weights": str(save) if save else None,
        "save_kind": "inference weights only; not a resumable training checkpoint" if save else None,
    }
