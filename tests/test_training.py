from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from fly_chess.config import Config
from fly_chess.neural.weights import load_weights
from fly_chess.storage.checkpoints import read_checkpoint
from fly_chess.training.controller import TrainingController
from fly_chess.training.replay import ReplayBuffer


def config():
    return Config(device="cpu", cpu_threads=1, network_channels=8, residual_blocks=1,
        mcts_simulations=1, games_per_generation=2, max_game_plies=2, batch_size=2,
        replay_buffer_size=8, updates_per_generation=2, training_inspection_interval=1, evaluation_enabled=False)


def latest(root):
    return root / "models/fly_latest.pt"


def test_real_selfplay_training_saves_complete_state_and_inference(tmp_path):
    with TrainingController(tmp_path, config()) as controller:
        before = {k: v.clone() for k, v in controller.model.state_dict().items()}
        result = controller.run()
        assert result["progress"]["training_steps"] == 2
        assert result["progress"]["completed_generations"] == 1
        assert result["progress"]["truncated_games"] == 2 and result["progress"]["draws"] == 0
        assert any(not torch.equal(before[k], v) for k, v in controller.model.state_dict().items())
        assert controller.optimizer.state
        metric = controller.metrics[-1]
        assert metric["loss"] == pytest.approx(metric["policy_loss"] + metric["value_loss"] + metric["regularization_loss"], abs=1e-6)
        assert metric["inspection"]["layers"] and metric["inspection"]["value_after"] != metric["inspection"]["value_before"]
        assert (tmp_path / "models/fly_generation_001.pt").is_file()
        assert not (tmp_path / "models/fly_best.pt").exists()
    checkpoint = read_checkpoint(latest(tmp_path), tmp_path)
    assert checkpoint["optimizer"]["state"] and checkpoint["scheduler"]["last_epoch"] == 2
    model, _ = load_weights(latest(tmp_path))
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"][k])


def test_resume_matches_uninterrupted_next_update_and_rng(tmp_path):
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    with TrainingController(continuous, config()) as controller:
        controller.run()
        expected = {k: v.clone() for k, v in controller.model.state_dict().items()}
        expected_losses = [row["loss"] for row in controller.metrics]
    def pause_after_first_update(controller):
        if controller.progress["training_steps"] == 1:
            controller.request_pause()
    with TrainingController(resumed, config()) as controller:
        result = controller.run(on_boundary=pause_after_first_update)
        assert result["progress"]["status"] == "paused"
        assert result["progress"]["updates_in_generation"] == 1
    with TrainingController(resumed, resume=latest(resumed)) as controller:
        assert controller.progress["training_steps"] == 1 and controller.optimizer.state
        controller.run()
        for key, value in controller.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert [row["loss"] for row in controller.metrics] == expected_losses


def test_mid_generation_resume_no_duplicate_games_and_stop(tmp_path):
    def stop_after_game(controller):
        if controller.progress["games_played"] == 1:
            controller.request_stop()
    with TrainingController(tmp_path, config()) as controller:
        result = controller.run(on_boundary=stop_after_game)
        assert result["progress"]["status"] == "stopped"
        assert result["progress"]["games_in_generation"] == 1
    with TrainingController(tmp_path, resume=latest(tmp_path)) as controller:
        controller.run()
        assert controller.progress["games_played"] == 2
        with ReplayBuffer(controller.replay_path, 8) as replay:
            assert replay.statistics()["truncated"] == 2 and len(replay) == 4


def test_archive_written_before_failed_checkpoint_recovers_once(tmp_path, monkeypatch):
    import fly_chess.training.controller as module
    with TrainingController(tmp_path, config()) as controller:
        original = module.atomic_torch_save
        def fail(payload, path):
            if payload["progress"]["games_played"]:
                raise OSError("simulated power loss before latest replacement")
            original(payload, path)
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "atomic_torch_save", fail)
            with pytest.raises(OSError):
                controller.run()
        assert read_checkpoint(latest(tmp_path), tmp_path)["progress"]["games_played"] == 0
    with TrainingController(tmp_path, resume=latest(tmp_path)) as controller:
        controller.run()
        assert controller.progress["games_played"] == 2
        assert len(list((tmp_path / "data/selfplay").glob("*.npz"))) == 2


def test_checkpoint_atomic_failure_retains_previous_bytes(tmp_path, monkeypatch):
    import fly_chess.storage.checkpoints as storage
    with TrainingController(tmp_path, config()) as controller:
        old = latest(tmp_path).read_bytes()
        def fail(*args):
            raise OSError("replace failed")
        monkeypatch.setattr(storage.os, "replace", fail)
        with pytest.raises(OSError):
            controller.save()
        assert latest(tmp_path).read_bytes() == old


def test_missing_corrupt_or_incompatible_checkpoints_and_manifest(tmp_path):
    with pytest.raises(ValueError, match="Cannot resume"):
        TrainingController(tmp_path, resume=tmp_path / "missing.pt")
    with TrainingController(tmp_path, config()):
        pass
    saved = latest(tmp_path).read_bytes()
    latest(tmp_path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="Cannot resume"):
        TrainingController(tmp_path, resume=latest(tmp_path))
    latest(tmp_path).write_bytes(saved)
    data = torch.load(latest(tmp_path), weights_only=True)
    data["training_schema"] = 99
    torch.save(data, latest(tmp_path))
    with pytest.raises(ValueError, match="schema"):
        TrainingController(tmp_path, resume=latest(tmp_path))
    latest(tmp_path).write_bytes(saved)
    manifest = torch.load(latest(tmp_path), weights_only=True)["replay"]
    (tmp_path / manifest["path"]).write_bytes(b"corrupted snapshot")
    with pytest.raises(ValueError, match="replay snapshot"):
        TrainingController(tmp_path, resume=latest(tmp_path))


def test_lock_and_existing_run_protection(tmp_path):
    with TrainingController(tmp_path, config()):
        with pytest.raises(RuntimeError, match="owns"):
            TrainingController(tmp_path, config())
    with pytest.raises(ValueError, match="already exists"):
        TrainingController(tmp_path, config())
    with pytest.raises(ValueError, match="configuration"):
        TrainingController(tmp_path, replace(config(), learning_rate=0.01), resume=latest(tmp_path))


def test_resume_in_fresh_python_process(tmp_path):
    with TrainingController(tmp_path, config()) as controller:
        controller.run(on_boundary=lambda c: c.request_pause() if c.progress["training_steps"] == 1 else None)
    result = subprocess.run([sys.executable, "-m", "fly_chess", "--workspace", str(tmp_path),
        "train", "--resume", str(latest(tmp_path))], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["progress"]["training_steps"] == 2 and report["progress"]["games_played"] == 2


def test_parallel_training_wave_and_manual_save(tmp_path):
    with TrainingController(tmp_path, replace(config(), selfplay_workers=2)) as controller:
        controller.request_save()
        controller.run()
        assert controller.progress["games_played"] == 2 and controller.progress["training_steps"] == 2
        assert not controller.save_event.is_set()


def test_failed_optimizer_commit_replays_exactly_once(tmp_path, monkeypatch):
    import fly_chess.training.controller as module
    reference, interrupted = tmp_path / "reference", tmp_path / "interrupted"
    with TrainingController(reference, config()) as trainer:
        trainer.run()
        expected = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    with TrainingController(interrupted, config()) as trainer:
        original = module.atomic_torch_save
        def fail(payload, path):
            if payload["progress"]["training_steps"] == 1:
                raise OSError("lost optimizer commit")
            original(payload, path)
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "atomic_torch_save", fail)
            with pytest.raises(OSError, match="optimizer commit"):
                trainer.run()
        assert read_checkpoint(latest(interrupted), interrupted)["progress"]["training_steps"] == 0
        with pytest.raises(RuntimeError, match="failed"):
            trainer.run()
    with TrainingController(interrupted, resume=latest(interrupted)) as trainer:
        trainer.run()
        assert trainer.progress["training_steps"] == 2
        assert len(trainer.metrics) == 2
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_rng_streams_restore_and_new_generation_continues(tmp_path):
    import random
    with TrainingController(tmp_path, config()) as trainer:
        trainer.run()
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(4)
        expected_sampler = trainer.rng.random()
    with TrainingController(tmp_path, resume=latest(tmp_path)) as trainer:
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
        assert trainer.rng.random() == expected_sampler
        trainer.run()
        assert trainer.progress["completed_generations"] == 2
        assert trainer.progress["training_steps"] == 4 and trainer.progress["next_game_index"] == 4
        assert (tmp_path / "models/fly_generation_002.pt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_resume(tmp_path):
    gpu_config = replace(config(), device="cuda", games_per_generation=1, updates_per_generation=1)
    with TrainingController(tmp_path, gpu_config) as trainer:
        trainer.run()
        expected = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        expected_rng = torch.cuda.get_rng_state().clone()
    with TrainingController(tmp_path, resume=latest(tmp_path)) as trainer:
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
        trainer.run()
        assert trainer.progress["training_steps"] == 2
