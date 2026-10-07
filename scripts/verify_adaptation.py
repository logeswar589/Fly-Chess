"""Matched-seed scripted-style experiment; no human rating or strength claim."""
import json
import math
from pathlib import Path
import time


def main():
    import numpy as np
    import torch
    from fly_chess.core.rules import ChessGame
    from fly_chess.game.opponent import OpponentSnapshot, features, observe
    from fly_chess.neural.inference import Evaluator
    from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
    from fly_chess.neural.runtime import seed_everything
    from fly_chess.search.mcts import MCTS, SearchSettings
    seed_everything(42, cpu_threads=2)
    model = PolicyValueNetwork(NetworkSpec(8, 1))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    evaluator = Evaluator(model, model_id='untrained-scripted-experiment')
    rows = []
    started = time.monotonic()
    for style, coefficients in (('capture_preference', [2., 0., 0.]), ('checks_and_center', [0., 2., 1.])):
        for seed in (17, 29):
            human = seed == 17
            for enabled in (False, True):
                game, profile = ChessGame(claim_draws=True), OpponentSnapshot()
                rng = np.random.default_rng(seed)
                baseline_loss = adapted_loss = 0.
                for ply in range(32):
                    if game.result:
                        break
                    if game.board.turn == human:
                        actions = game.actions()
                        baseline = evaluator.evaluate([game])[0].policy[actions]
                        prediction = profile.predict(game, actions, baseline)
                        logits = features(game, actions) @ coefficients
                        distribution = np.exp(logits-logits.max())
                        distribution /= distribution.sum()
                        index = int(rng.choice(len(actions), p=distribution))
                        baseline_loss -= math.log(max(float(baseline[index]), 1e-12))
                        adapted_loss -= math.log(max(float(prediction[index]), 1e-12))
                        action = int(actions[index])
                        profile = observe(profile, game, action, baseline)
                        game.push_action(action)
                    else:
                        settings = SearchSettings(simulations=8, opponent_weight=profile.strength if enabled else 0.)
                        result = MCTS(evaluator, settings, seed=seed*100+ply).search(game,
                            opponent_prior=profile.predict if enabled and profile.strength else None)
                        assert result.move in game.board.legal_moves
                        game.push(result.move)
                rows.append({'style': style, 'seed': seed, 'human_color': 'white' if human else 'black',
                    'adaptation': enabled, 'plies': len(game.board.move_stack),
                    'status': 'completed' if game.result else 'truncated',
                    'human_result': game.result.for_player(human) if game.result else None,
                    'decisions': profile.decisions, 'mean_baseline_log_loss': baseline_loss/max(1, profile.decisions),
                    'mean_adapted_log_loss': adapted_loss/max(1, profile.decisions), 'final_influence': profile.strength})
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    report = {'model': evaluator.model_id, 'simulations': 8, 'max_plies': 32, 'matched_pairs': 4,
        'seconds': time.monotonic()-started, 'main_weights_unchanged': True, 'games': rows,
        'conclusion': 'Small scripted diagnostic, not evidence of chess improvement. Adaptation remains experimental and off by default.'}
    target = Path('logs/adaptation-experiment.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({'report': str(target.resolve()), 'seconds': report['seconds'],
        'games': len(rows), 'completed': sum(r['status'] == 'completed' for r in rows),
        'main_weights_unchanged': True, 'conclusion': report['conclusion']}, indent=2))


if __name__ == '__main__':
    main()
