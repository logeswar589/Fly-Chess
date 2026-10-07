from fly_chess.neural.inference import Evaluator
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import load_weights
from fly_chess.search.mcts import MCTS, SearchSettings


def diagnose_search(config, game, *, weights=None, seconds=None, self_play=False, inspect=False):
    seed_everything(config.seed, deterministic=config.deterministic, cpu_threads=config.cpu_threads)
    selection = select_device(config.device)
    if weights:
        model, model_id = load_weights(weights, device=selection.device)
    else:
        model = PolicyValueNetwork(NetworkSpec(config.network_channels, config.residual_blocks)).to(selection.device)
        model_id = f"untrained-seed-{config.seed}"
    search = MCTS(Evaluator(model, model_id=model_id), SearchSettings(
        simulations=config.mcts_simulations, cpuct=config.mcts_cpuct, max_seconds=seconds,
        max_depth=config.mcts_max_depth, max_nodes=config.mcts_max_nodes, use_mcts=config.mcts_enabled,
    ), seed=config.seed)
    result = search.search(game, self_play=self_play, request_id="cli-search",
                           game_id="diagnostic", tree_edges=128 if inspect else 0)
    return {"phase": 3, "device": str(selection.device), "device_reason": selection.reason,
            "weights_source": str(weights) if weights else "random initialization; not trained",
            "policy_sum": float(result.policy.sum()), **result.snapshot}
