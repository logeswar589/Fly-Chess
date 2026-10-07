from dataclasses import replace
import json

import pytest
import torch

from fly_chess.config import Config
from fly_chess.evaluation.arena import ArenaSettings, evaluate_candidate, play_match
from fly_chess.evaluation.statistics import summarize
from fly_chess.neural.inference import Evaluator
from fly_chess.neural.network import NetworkSpec, PolicyValueNetwork
from fly_chess.neural.runtime import seed_everything
from fly_chess.neural.weights import save_weights
from fly_chess.storage.checkpoints import file_hash
from fly_chess.training.controller import TrainingController


def game(score=0.5, *, candidate_white=True, status="completed", opening=None, seed=1):
    return {"candidate_white": candidate_white, "candidate_score": score if status == "completed" else None,
            "status": status, "termination": "test_fixture", "moves": [], "opening": opening or [], "seed": seed}


def games(scores):
    return [game(score, candidate_white=i % 2 == 0) for i, score in enumerate(scores)]


def test_paired_confidence_and_relative_rating():
    result = summarize(games([1, 1] * 25 + [1, 0] * 25), requested_pairs=50, min_pairs=20, alpha=0.05)
    assert result["wins"] == 75 and result["losses"] == 25
    assert result["score_rate"] == 0.75 and result["completed_pairs"] == 50
    assert result["score_interval"][0] > 0.5 and result["promote"]
    assert result["relative_elo_difference"] == pytest.approx(190.8485, abs=0.001)
    # This uses 50 independent PAIRS in the bound, not 100 allegedly independent games.
    assert result["score_interval"][0] == pytest.approx(0.55794, abs=0.001)


def test_draws_small_samples_extremes_and_truncations():
    draws = summarize(games([0.5] * 100), requested_pairs=50, min_pairs=20, alpha=0.05)
    assert draws["draws"] == 100 and not draws["promote"] and draws["relative_elo_difference"] == 0
    small = summarize(games([1, 1]), requested_pairs=1, min_pairs=20, alpha=0.05)
    assert not small["promotion_eligible"] and small["rating_status"] == "unrated"
    for score in (0, 1):
        extreme = summarize(games([score] * 100), requested_pairs=50, min_pairs=20, alpha=0.05)
        assert extreme["relative_elo_difference"] is None
        assert extreme["promote"] == bool(score)
    records = games([1] * 100)
    records[0] = game(status="truncated")
    result = summarize(records, requested_pairs=50, min_pairs=20, alpha=0.05)
    assert result["truncated"] == 1 and result["draws"] == 0 and not result["promote"]


def make_weights(tmp_path):
    seed_everything(7, cpu_threads=1)
    model = PolicyValueNetwork(NetworkSpec(8, 1))
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    save_weights(model, first, model_id="baseline")
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    save_weights(model, second, model_id="candidate")
    return first, second


def runner(score=0.5, calls=None):
    def run(candidate, incumbent, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return game(score, candidate_white=kwargs["candidate_white"], opening=kwargs["opening"], seed=kwargs["seed"])
    return run


def test_baseline_rejection_and_only_successful_promotion_changes_best(tmp_path):
    first, second = make_weights(tmp_path)
    settings = ArenaSettings(pairs=50)
    baseline = evaluate_candidate(tmp_path, first, evaluation_id="baseline", settings=settings, device="cpu")
    assert baseline["status"] == "baseline" and baseline["statistics"] is None
    path = tmp_path / "models/fly_best.pt"
    original_hash = file_hash(path)
    rejected = evaluate_candidate(tmp_path, second, evaluation_id="drawn", settings=settings, device="cpu", match_runner=runner())
    assert not rejected["promoted"] and file_hash(path) == original_hash
    promoted = evaluate_candidate(tmp_path, second, evaluation_id="won", settings=settings, device="cpu", match_runner=runner(1))
    assert promoted["promoted"] and file_hash(path) != original_hash
    def forbidden(*args, **kwargs):
        raise AssertionError("An already promoted evaluation must not run games again")
    again = evaluate_candidate(tmp_path, second, evaluation_id="won", settings=settings, device="cpu", match_runner=forbidden)
    assert again == promoted


def test_interrupted_evaluation_resumes_same_pairs_and_preserves_best(tmp_path):
    first, second = make_weights(tmp_path)
    settings = ArenaSettings(pairs=2)
    evaluate_candidate(tmp_path, first, evaluation_id="initial", settings=settings, device="cpu")
    best_hash = file_hash(tmp_path / "models/fly_best.pt")
    calls = []
    interrupted = evaluate_candidate(tmp_path, second, evaluation_id="partial", settings=settings, device="cpu",
        match_runner=runner(calls=calls), cancel=lambda: len(calls) >= 1)
    assert interrupted["status"] == "interrupted" and len(interrupted["games"]) == 1
    resumed = evaluate_candidate(tmp_path, second, evaluation_id="partial", settings=settings, device="cpu", match_runner=runner(calls=calls))
    assert resumed["status"] == "complete" and len(calls) == 4
    assert [c["candidate_white"] for c in calls] == [True, False, True, False]
    assert calls[0]["opening"] == calls[1]["opening"] and calls[0]["seed"] == calls[1]["seed"]
    assert file_hash(tmp_path / "models/fly_best.pt") == best_hash


def test_failed_promotion_write_recovers_from_finished_journal(tmp_path, monkeypatch):
    import fly_chess.evaluation.arena as arena
    first, second = make_weights(tmp_path)
    settings = ArenaSettings(pairs=50)
    evaluate_candidate(tmp_path, first, evaluation_id="initial", settings=settings, device="cpu")
    original_hash = file_hash(tmp_path / "models/fly_best.pt")
    with monkeypatch.context() as scoped:
        def fail(*args):
            raise OSError("promotion replacement failed")
        scoped.setattr(arena, "atomic_torch_save", fail)
        with pytest.raises(OSError):
            evaluate_candidate(tmp_path, second, evaluation_id="winner", settings=settings, device="cpu", match_runner=runner(1))
    assert file_hash(tmp_path / "models/fly_best.pt") == original_hash
    def forbidden(*args, **kwargs):
        raise AssertionError("Finished games should not be repeated")
    result = evaluate_candidate(tmp_path, second, evaluation_id="winner", settings=settings, device="cpu", match_runner=forbidden)
    assert result["promoted"] and file_hash(tmp_path / "models/fly_best.pt") != original_hash


def test_reusing_identity_with_different_settings_fails(tmp_path):
    first, _ = make_weights(tmp_path)
    settings = ArenaSettings(pairs=1)
    evaluate_candidate(tmp_path, first, evaluation_id="baseline", settings=settings, device="cpu")
    with pytest.raises(ValueError, match="different weights or settings"):
        evaluate_candidate(tmp_path, first, evaluation_id="baseline", settings=replace(settings, pairs=2), device="cpu")


def test_real_arena_rules_and_ply_limit():
    seed_everything(42, cpu_threads=1)
    evaluator = Evaluator(PolicyValueNetwork(NetworkSpec(8, 1)))
    settings = ArenaSettings(pairs=1, simulations=1, max_plies=1)
    result = play_match(evaluator, evaluator, candidate_white=True, opening=[], settings=settings, seed=1)
    assert result["status"] == "truncated" and result["candidate_score"] is None
    result = play_match(evaluator, evaluator, candidate_white=False,
        opening=["f2f3", "e7e5", "g2g4", "d8h4"], settings=settings, seed=1)
    assert result["status"] == "completed" and result["candidate_score"] == 1


def test_training_evaluates_each_generation_without_fake_elo(tmp_path):
    config = Config(device="cpu", cpu_threads=1, network_channels=8, residual_blocks=1,
        mcts_simulations=1, games_per_generation=1, max_game_plies=2, batch_size=2,
        replay_buffer_size=8, updates_per_generation=1, evaluation_pairs=1,
        evaluation_simulations=1, evaluation_max_plies=2)
    with TrainingController(tmp_path, config) as trainer:
        trainer.run()
        assert trainer.evaluations[0]["status"] == "baseline"
        assert torch.load(tmp_path / "models/fly_best.pt", weights_only=True)["training_schema"] == 1
        before = file_hash(tmp_path / "models/fly_best.pt")
        trainer.run()
        assert trainer.evaluations[1]["status"] == "complete"
        assert trainer.evaluations[1]["statistics"]["truncated"] == 2
        assert not trainer.evaluations[1]["promoted"]
        assert trainer.progress["estimated_elo"] is None
        assert file_hash(tmp_path / "models/fly_best.pt") == before
    with TrainingController(tmp_path, resume=tmp_path / "models/fly_latest.pt") as trainer:
        assert len(trainer.evaluations) == 2
        assert trainer.progress["completed_generations"] == 2
