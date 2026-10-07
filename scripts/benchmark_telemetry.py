"""Local instrumentation overhead, not a chess-strength benchmark."""
from pathlib import Path
import json
import statistics
import time
import tracemalloc


def main():
    import torch
    from fly_chess.core.rules import ChessGame
    from fly_chess.neural.runtime import seed_everything, select_device
    from fly_chess.neural.network import PolicyValueNetwork
    from fly_chess.neural.inference import Evaluator
    from fly_chess.neural.inspection import InspectionRequest
    from fly_chess.search.mcts import MCTS, SearchSettings
    from fly_chess.storage.games import load_game
    from fly_chess.training.optimizer import train_step
    seed_everything(42)
    device = select_device('auto').device
    model = PolicyValueNetwork().to(device)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    evaluator = Evaluator(model)
    game = ChessGame()
    settings = SearchSettings(simulations=32)
    source = next((Path.cwd()/'logs').glob('gui-training-*/data/selfplay/*.npz'))
    samples = load_game(source).samples[:8]
    report = {'device': str(device), 'network': '32 channels / 2 residual blocks', 'trials': 5}
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    def measure(function):
        function()
        times = []
        for _ in range(5):
            sync()
            started = time.perf_counter()
            function()
            sync()
            times.append(time.perf_counter()-started)
        tracemalloc.start()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        function()
        sync()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {'median_seconds': statistics.median(times), 'python_peak_bytes_during_call': peak,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
    for enabled in (False, True):
        model.load_state_dict(initial)
        def search():
            evaluator.model.eval()
            if enabled:
                evaluator.evaluate([game], inspection=InspectionRequest('benchmark', selected_module='stem', max_values=65536, values_per_tensor=65536))
            return MCTS(evaluator, settings).search(game, tree_edges=512 if enabled else 0,
                on_progress=(lambda data: None) if enabled else None)
        report['search_on' if enabled else 'search_off'] = measure(search)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        report['update_on' if enabled else 'update_off'] = measure(lambda: train_step(model, optimizer, samples,
            regularization=.0001, gradient_clip=1., inspect=enabled, activation_module='stem' if enabled else None))
    for kind in ('search', 'update'):
        report[kind+'_overhead_percent'] = 100*(report[kind+'_on']['median_seconds']/report[kind+'_off']['median_seconds']-1)
    report['notes'] = 'Warm local kernels; update-on samples every step (GUI default every 8). Memory excludes preexisting Python allocations; CUDA reports total allocated peak. No rendering or disk/checkpoint I/O.'
    path = Path('logs/telemetry-benchmark.json')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
