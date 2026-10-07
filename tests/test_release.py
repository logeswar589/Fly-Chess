from pathlib import Path
import numpy as np
import torch

from fly_chess.core.rules import ChessGame
from fly_chess.core.encoding import encode_board
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork, module_inventory
from fly_chess.neural.weights import save_weights
from fly_chess.neural.runtime import seed_everything
from fly_chess.ui.worker import BrainWorker


def request(worker, path, token, module='stem'):
    assert worker.submit(token=token, game=ChessGame(), human_color=True, weights=path, module=module)
    worker.future.result(timeout=30)
    message = worker.drain()[-1]
    assert message['kind'] == 'inspection', message
    return message


def test_same_match_freezes_checkpoint_and_new_match_reloads_replacement(tmp_path):
    seed_everything(8)
    model = PolicyValueNetwork(NetworkSpec(8, 1))
    path = tmp_path/'latest.pt'
    save_weights(model, path, model_id='before-training')
    worker = BrainWorker()
    try:
        first = request(worker, path, ('game-one', 0))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.01)
        save_weights(model, path, model_id='after-training')
        same = request(worker, path, ('game-one', 1))
        new = request(worker, path, ('game-two', 0))
        assert first['evaluation'].model_id == same['evaluation'].model_id == 'before-training'
        np.testing.assert_array_equal(first['evaluation'].policy, same['evaluation'].policy)
        assert new['evaluation'].model_id == 'after-training'
        assert not np.array_equal(new['evaluation'].policy, first['evaluation'].policy)
    finally:
        worker.close()


def test_every_module_and_tensor_reachable_with_exact_weights_and_input(tmp_path):
    seed_everything(19)
    model = PolicyValueNetwork(NetworkSpec(8, 1))
    path = tmp_path/'model.pt'
    save_weights(model, path, model_id='inventory-test')
    worker = BrainWorker()
    try:
        for index, entry in enumerate(module_inventory(model)):
            message = request(worker, path, ('inventory', index), entry['name'])
            inspection = message['evaluation'].inspection
            selected = {k: v for k, v in inspection['tensors'].items() if k.rsplit(':', 1)[0] == entry['name']}
            assert selected and all(v['sample_values'] for v in selected.values())
            assert sum(len(v['sample_values']) for v in inspection['tensors'].values()) <= 65536
            np.testing.assert_array_equal(message['input']['values'], encode_board(ChessGame().board))
            module = dict(model.named_modules())['' if entry['name'] == '<network>' else entry['name']]
            parameters = dict(module.named_parameters(recurse=False))
            for weight in message['weights']:
                np.testing.assert_array_equal(weight['values'], parameters[weight['name']].detach().reshape(-1)[:4096].numpy())
            assert inspection['fen'] == ChessGame().board.fen()
            assert inspection['model_id'] == 'inventory-test'
        assert all(not module._forward_hooks for module in worker.evaluator.model.modules())
    finally:
        worker.close()
