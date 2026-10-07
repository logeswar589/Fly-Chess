"""Explicit supervised human-data candidate training, separate from self-play replay."""
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
import numpy as np
import torch

from fly_chess.game.human_data import load_human_dataset
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import SCHEMAS, load_weights
from fly_chess.storage.checkpoints import atomic_torch_save, file_hash
from fly_chess.storage.workspace import Workspace
from fly_chess.training.optimizer import train_step
from fly_chess.evaluation.arena import ArenaSettings, atomic_json, evaluate_candidate


def train_human_candidate(root, base, config, *, epochs=1, max_positions=4096, learning_rate=.0001,
                          cancel=lambda: False, on_progress=None):
    if type(epochs) is not int or not 1 <= epochs <= 8:
        raise ValueError('Human training epochs must be in [1, 8]')
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= .001:
        raise ValueError('Human training learning rate must be in (0, .001]')
    root, base = Path(root).resolve(), Path(base).resolve()
    Workspace(root).initialize()
    with FileLock(str(root/'models/.training.lock'), timeout=0):
        dataset = load_human_dataset(root, max_positions=max_positions)
        seed_everything(config.seed, deterministic=config.deterministic, cpu_threads=config.cpu_threads)
        parent_hash = file_hash(base)
        model, parent_id = load_weights(base, device=select_device(config.device).device)
        if file_hash(base) != parent_hash:
            raise ValueError('Base checkpoint changed while loading; retry with an immutable checkpoint.')
        run_id = 'human-'+uuid4().hex
        candidate = root / f'models/human_candidates/{run_id}.pt'
        journal = root / f'logs/human_training/{run_id}.json'
        provenance = {'schema': 1, 'run_id': run_id, 'parent_model_id': parent_id,
            'parent_sha256': parent_hash, 'objective': 'human legal-move cross entropy + outcome MSE + L2',
            'epochs': epochs, 'learning_rate': learning_rate, 'seed': config.seed,
            'positions': len(dataset.samples), 'sample_keys': dataset.keys,
            'excluded_incomplete': dataset.excluded_incomplete, 'sources': dataset.sources}
        dataset_path = root / f'data/human_training/{run_id}.json'
        atomic_json(provenance, dataset_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rng = np.random.default_rng(config.seed)
        report = {'run_id': run_id, 'status': 'training', 'candidate': str(candidate),
            'dataset': str(dataset_path), 'positions': len(dataset.samples), 'epochs': epochs,
            'updates': [], 'evaluation_id': run_id, 'evaluation': None}
        def publish():
            atomic_json(report, journal)
            if on_progress:
                on_progress({**report, 'updates': [dict(row) for row in report['updates']]})
        publish()
        for epoch in range(epochs):
            order = rng.permutation(len(dataset.samples))
            for start in range(0, len(order), config.batch_size):
                if cancel():
                    break
                samples = [dataset.samples[int(i)] for i in order[start:start+config.batch_size]]
                update = train_step(model, optimizer, samples, regularization=config.weight_decay,
                    gradient_clip=config.gradient_clip)
                report['updates'].append({**update, 'epoch': epoch+1, 'step': len(report['updates'])+1})
                publish()
            if cancel():
                break
        report['status'] = 'interrupted_training' if cancel() else 'evaluating'
        payload = {**SCHEMAS, 'spec': asdict(model.spec), 'model_id': run_id,
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'human_training': {k: v for k, v in provenance.items() if k != 'sources'},
            'human_training_status': 'partial' if cancel() else 'complete',
            'human_dataset_sha256': file_hash(dataset_path), 'human_updates': report['updates']}
        atomic_torch_save(payload, candidate)
        publish()
        if cancel():
            return report
        settings = ArenaSettings(pairs=config.evaluation_pairs, min_pairs=config.evaluation_min_pairs,
            simulations=config.evaluation_simulations, max_plies=config.evaluation_max_plies, seed=config.seed)
        # If there is no incumbent, establish the unchanged parent as baseline first.
        # The human-trained candidate must still compete through the normal gate.
        if not (root/'models/fly_best.pt').exists():
            if file_hash(base) != parent_hash:
                raise ValueError('Base checkpoint changed before baseline evaluation; candidate is preserved, best is unchanged.')
            evaluate_candidate(root, base, evaluation_id=run_id+'-base', settings=settings, device=config.device, cancel=cancel)
        report['evaluation'] = evaluate_candidate(root, candidate, evaluation_id=run_id,
            settings=settings, device=config.device, cancel=cancel)
        report['status'] = 'complete' if report['evaluation']['status'] == 'complete' else 'interrupted_evaluation'
        publish()
        return report
