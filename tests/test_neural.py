import json

import chess
import numpy as np
import pytest
import torch
from torch import nn

from fly_chess.cli import main
from fly_chess.core.actions import ACTION_SIZE, encode_move, legal_mask
from fly_chess.core.encoding import encode_board
from fly_chess.core.rules import ChessGame
from fly_chess.neural.inference import Evaluator, PolicyAgent, masked_policy, position_id
from fly_chess.neural.inspection import ActivationCapture, InspectionRequest
from fly_chess.neural.network import NetworkSpec, PolicyHead, PolicyValueNetwork, module_inventory
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import load_weights, save_weights


@pytest.fixture
def model():
    seed_everything(31, cpu_threads=1)
    return PolicyValueNetwork(NetworkSpec(8, 1))


def states():
    board = chess.Board()
    first = encode_board(board)
    board.push_uci("e2e4")
    return torch.from_numpy(np.stack([first, encode_board(board)]))


def test_forward_shapes_and_single_batch_consistency(model):
    model.eval()
    with torch.inference_mode():
        logits, values = model(states())
        one_logits, one_value = model(states()[:1])
    assert logits.shape == (2, ACTION_SIZE)
    assert values.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert (values.abs() <= 1).all()
    torch.testing.assert_close(one_logits[0], logits[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(one_value[0], values[0], atol=1e-5, rtol=1e-5)


def test_policy_head_uses_documented_action_order():
    head = PolicyHead(1)
    with torch.no_grad():
        head.projection.weight.fill_(1)
        head.projection.bias.copy_(torch.arange(320, dtype=torch.float32))
    board = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) * 1000
    output = head(board)[0]
    for move in ("e2e4", "a7a8q", "h2h1n", "e1g1"):
        decoded = chess.Move.from_uci(move)
        index = encode_move(decoded)
        assert output[index].item() == decoded.from_square * 1000 + index % 320


def test_mask_handles_extremes_terminal_rows_and_illegal_nan():
    logits = torch.full((2, ACTION_SIZE), float("nan"))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 10:12] = True
    logits[0, 10:12] = torch.tensor([10000., 9999.])
    result = masked_policy(logits, mask)
    assert torch.isfinite(result).all()
    assert torch.all(result[~mask] == 0)
    assert result[0].sum().item() == pytest.approx(1)
    assert result[0, 10].item() == pytest.approx(0.7310586)
    assert result[1].sum() == 0
    logits[0, 10] = float("inf")
    with pytest.raises(ValueError, match="Non-finite"):
        masked_policy(logits, mask)


def test_mixed_batch_terminal_exact_value_and_empty_input(model):
    evaluator = Evaluator(model)
    game = ChessGame()
    mate = ChessGame.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    draw = ChessGame.from_fen("7k/8/6K1/8/8/8/8/8 w - - 0 1")
    results = evaluator.evaluate([mate, game, draw])
    assert results[0].terminal and results[0].value == -1
    assert results[2].terminal and results[2].value == 0
    assert not results[0].policy.any() and not results[2].policy.any()
    assert results[1].policy.sum() == pytest.approx(1)
    assert not results[1].policy[~legal_mask(game.board)].any()
    assert evaluator.evaluate([]) == []
    assert PolicyAgent(evaluator).choose_move(mate) is None


def test_terminal_bypasses_network_and_bad_values_raise(model, monkeypatch):
    evaluator = Evaluator(model)
    def broken_forward(x):
        raise AssertionError("Terminal state should not invoke model")
    monkeypatch.setattr(model, "forward", broken_forward)
    evaluator.evaluate([ChessGame.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")])
    monkeypatch.setattr(model, "forward", lambda x: (torch.zeros(len(x), ACTION_SIZE), torch.full((len(x),), float("nan"))))
    with pytest.raises(ValueError, match="invalid values"):
        evaluator.evaluate([ChessGame()])


def test_actual_optimizer_step_has_finite_gradients_and_updates_both_heads(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    logits, values = model(states())
    targets = torch.tensor([encode_move(chess.Move.from_uci("e2e4")), encode_move(chess.Move.from_uci("e7e5"))])
    loss = nn.functional.cross_entropy(logits, targets) + nn.functional.mse_loss(values, torch.tensor([1., -1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    for prefix in ("stem", "trunk", "policy", "value"):
        assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters() if name.startswith(prefix))


def test_weights_roundtrip_and_reject_schema(model, tmp_path):
    path = tmp_path / "weights.pt"
    save_weights(model, path, model_id="test-only")
    rng_before = torch.get_rng_state().clone()
    restored, model_id = load_weights(path)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert model_id == "test-only" and not restored.training
    torch.testing.assert_close(model(states()), restored(states()), rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    payload["encoding_schema"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_weights(path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Cannot load"):
        load_weights(path)


def test_failed_save_preserves_previous_file(model, tmp_path, monkeypatch):
    import fly_chess.neural.weights as storage
    path = tmp_path / "weights.pt"
    save_weights(model, path)
    previous = path.read_bytes()
    def fail(*args):
        raise OSError("simulated failed replace")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError):
        save_weights(model, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_inspection_matches_activations_is_bounded_and_does_not_change_outputs(model):
    evaluator = Evaluator(model)
    game = ChessGame()
    baseline = evaluator.evaluate([game])[0]
    inspected = evaluator.evaluate([game], inspection=InspectionRequest("req-7", max_values=100, values_per_tensor=16))[0]
    np.testing.assert_array_equal(baseline.policy, inspected.policy)
    assert baseline.value == inspected.value
    snapshot = inspected.inspection
    assert snapshot["position_id"] == position_id(game)
    assert snapshot["request_id"] == "req-7"
    samples = snapshot["tensors"]
    assert sum(value["sampled_elements"] for value in samples.values()) <= 100
    assert {key.rsplit(":", 1)[0] for key in samples} == {row["name"] for row in module_inventory(model)}
    with torch.no_grad():
        expected = model.stem[0](torch.from_numpy(encode_board(game.board)).unsqueeze(0)).flatten()[:16].tolist()
    assert samples["stem.0:0"]["sample_values"] == pytest.approx(expected)
    json.dumps(snapshot, allow_nan=False)  # Only serializable detached data leaves worker.
    assert all(not module._forward_hooks for module in model.modules())
    selected = evaluator.evaluate([game], inspection=InspectionRequest("req-8", selected_module="policy"))[0]
    assert selected.inspection["tensors"]["policy:0"]["sampled_elements"] > 0
    assert selected.inspection["tensors"]["stem.0:0"]["sampled_elements"] == 0


def test_inspection_cleanup_on_error(model):
    with pytest.raises(RuntimeError):
        with ActivationCapture(model, InspectionRequest("bad-run"), {}):
            raise RuntimeError("failure")
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(ValueError, match="Unknown module"):
        with ActivationCapture(model, InspectionRequest("bad-module", selected_module="missing"), {}):
            pass


def test_seeded_initialization_and_legal_sampling_reproduce():
    seed_everything(123, cpu_threads=1)
    first = PolicyValueNetwork(NetworkSpec(8, 1))
    seed_everything(123, cpu_threads=1)
    second = PolicyValueNetwork(NetworkSpec(8, 1))
    torch.testing.assert_close(first(states()), second(states()), rtol=0, atol=0)
    agents = [PolicyAgent(Evaluator(model), seed=99) for model in (first, second)]
    games = [ChessGame(), ChessGame()]
    for _ in range(16):
        moves = [agent.choose_move(game, temperature=0.7) for agent, game in zip(agents, games)]
        assert moves[0] == moves[1]
        if moves[0] is None:
            break
        for game, move in zip(games, moves):
            game.push(move)


def test_device_fallback_and_explicit_cuda_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").device.type == "cpu"
    assert select_device("cpu").device.type == "cpu"
    with pytest.raises(RuntimeError, match="not usable"):
        select_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def fail(*args, **kwargs):
        raise RuntimeError("driver failure")
    monkeypatch.setattr(torch, "ones", fail)
    assert "driver failure" in select_device().reason


def test_neural_cli_reports_untrained_and_reloads(tmp_path, capsys):
    path = tmp_path / "test.pt"
    base = ["--workspace", str(tmp_path), "neural-diagnose"]
    assert main([*base, "--save-weights", str(path)]) == 0
    original = json.loads(capsys.readouterr().out)
    assert "not trained" in original["weights_source"]
    assert original["policy_sum"] == pytest.approx(1)
    assert main([*base, "--weights", str(path), "--inspect"]) == 0
    restored = json.loads(capsys.readouterr().out)
    assert original["top_moves"] == restored["top_moves"]
    assert original["value"] == restored["value"]
    assert restored["inspection"]["model_id"] == original["model_id"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_backward_and_reload(model, tmp_path):
    selection = select_device("auto")
    assert selection.device.type == "cuda"
    model = model.to(selection.device)
    batch = states().to(selection.device)
    first = model(batch)
    second = model(batch)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    logits, values = first
    loss = nn.functional.cross_entropy(logits, torch.tensor([1, 2], device="cuda")) + values.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    path = tmp_path / "gpu-weights.pt"
    save_weights(model, path)
    restored, _ = load_weights(path, device="cuda")
    torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)
    result = Evaluator(restored).evaluate([ChessGame()], inspection=InspectionRequest("cuda-inspect"))[0]
    assert result.policy.sum() == pytest.approx(1)
    assert result.inspection["tensors"]["stem.0:0"]["sampled_elements"] > 0
