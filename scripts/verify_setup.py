"""Read-only installation smoke check; does not create or train checkpoints."""
import argparse
import json
from pathlib import Path
import sys


def verify(require_cuda=False):
    import chess
    import numpy as np
    import torch
    import pygame
    import filelock
    import fly_chess
    from fly_chess.core.rules import ChessGame
    from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
    from fly_chess.neural.runtime import select_device, seed_everything
    from fly_chess.neural.inference import Evaluator
    from fly_chess.neural.inspection import InspectionRequest
    from fly_chess.web.server import Server

    seed_everything(42)
    device = select_device('cuda' if require_cuda else 'auto')
    model = PolicyValueNetwork(NetworkSpec(8, 1)).to(device.device)
    result = Evaluator(model).evaluate([ChessGame()], inspection=InspectionRequest('setup', brain=True))[0]
    assert np.isfinite(result.policy).all() and abs(float(result.policy.sum())-1) < 1e-5
    assert result.inspection['brain'] and result.inspection['connections']
    package = Path(fly_chess.__file__).parent
    for relative in ('web/static/index.html', 'web/static/app.js', 'web/static/style.css',
                     'ui/assets/Sans.ttf', 'ui/assets/Mono.ttf'):
        assert (package/relative).is_file(), 'Missing installed asset: '+relative
    pygame.font.init()
    try:
        assert pygame.font.Font(str(package/'ui/assets/Sans.ttf'), 16).render('Fly', True, 'white').get_width() > 0
    finally:
        pygame.font.quit()
    return {'status': 'passed', 'python': sys.version.split()[0], 'torch': torch.__version__,
            'device': str(device.device), 'device_detail': device.reason,
            'package': str(package), 'legal_moves': len(list(chess.Board().legal_moves)),
            'browser_assets': 'present', 'neural_inference': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.require_cuda), indent=2))
