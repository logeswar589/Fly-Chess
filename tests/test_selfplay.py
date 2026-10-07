from dataclasses import replace
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from fly_chess.config import Config
from fly_chess.core.actions import ACTION_SIZE, encode_move
from fly_chess.core.rules import ChessGame
from fly_chess.search.mcts import SearchResult, SearchSettings
from fly_chess.selfplay.game import play_game
from fly_chess.selfplay.workers import game_seed, run_batch
from fly_chess.storage.games import load_game, save_game
from fly_chess.training.replay import ReplayBuffer


class ScriptedSearch:
    settings = SearchSettings()
    evaluator = type("Identity", (), {"model_id": "test-script"})()

    def __init__(self, moves, on_move=None):
        self.moves = iter(moves)
        self.on_move = on_move

    def search(self, game, **kwargs):
        import chess
        action = encode_move(chess.Move.from_uci(next(self.moves)))
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[action] = 1
        if self.on_move:
            self.on_move()
        return SearchResult(action, policy, 0.2, 1, "simulation_limit", "mcts_visits", {"network_value": 0.2})


def mate_record(game_id="mate"):
    return play_game(ScriptedSearch(["f2f3", "e7e5", "g2g4", "d8h4"]), game_id=game_id, seed=42)


def test_complete_game_correct_targets_for_both_players():
    record = mate_record()
    assert record.status == "completed" and record.termination == "checkmate"
    assert record.white_result == -1
    assert [s.target for s in record.samples] == [-1, 1, -1, 1]
    record.validate()


def test_truncation_and_abort_are_not_rule_draws():
    record = play_game(ScriptedSearch(["e2e4", "e7e5"]), game_id="limit", seed=2, max_plies=2)
    assert record.status == "truncated" and record.termination == "ply_limit"
    assert [s.target for s in record.samples] == [0, 0]
    stopped = [False]
    record = play_game(ScriptedSearch(["e2e4"], on_move=lambda: stopped.__setitem__(0, True)),
                       game_id="abort", seed=3, cancel=lambda: stopped[0])
    assert record.status == "aborted" and record.white_result is None
    assert record.samples[0].target is None


def test_initial_terminal_and_prefix_history():
    game = ChessGame.from_fen("7k/8/6K1/8/8/8/8/8 w - - 0 1")
    record = play_game(ScriptedSearch([]), game_id="draw", seed=4, start=game)
    assert record.status == "completed" and not record.samples and record.white_result == 0
    game = ChessGame()
    game.push("f2f3")
    record = play_game(ScriptedSearch(["e7e5", "g2g4", "d8h4"]), game_id="prefix", seed=5, start=game)
    assert record.prefix_moves == ["f2f3"]
    assert [s.target for s in record.samples] == [1, -1, 1]


def test_archive_roundtrip_and_malformed_data(tmp_path):
    path = tmp_path / "game.npz"
    record = mate_record()
    save_game(record, path)
    loaded = load_game(path)
    assert loaded.metadata() == record.metadata()
    for original, restored in zip(record.samples, loaded.samples):
        np.testing.assert_array_equal(original.state, restored.state)
        np.testing.assert_array_equal(original.policy, restored.policy)
        assert original.target == restored.target
    path.write_bytes(b"not-an-archive")
    with pytest.raises(ValueError, match="Cannot load"):
        load_game(path)
    record.samples[0].target = 1
    with pytest.raises(ValueError, match="perspective"):
        save_game(record, path)


def test_replay_capacity_restart_dedup_and_seeded_sampling(tmp_path):
    path = tmp_path / "replay.sqlite3"
    with ReplayBuffer(path, 5) as replay:
        assert replay.add_game(mate_record("one"))
        assert replay.add_game(mate_record("two"))
        assert len(replay) == 5
        assert not replay.add_game(mate_record("one"))
        samples = replay.sample(5, np.random.default_rng(8))
        assert {(r["game_id"], r["ply"]) for r in samples} == {("one", 3), *(('two', i) for i in range(4))}
    with ReplayBuffer(path, 5) as replay:
        restored = replay.sample(5, np.random.default_rng(8))
        assert [(r["game_id"], r["ply"]) for r in samples] == [(r["game_id"], r["ply"]) for r in restored]
        assert replay.statistics()["black_wins"] == 2
        conflict = mate_record("one")
        conflict.seed = 99
        with pytest.raises(ValueError, match="different data"):
            replay.add_game(conflict)
        with pytest.raises(ValueError, match="batch_size"):
            replay.sample(6, np.random.default_rng())
    with ReplayBuffer(path, 2) as replay:
        assert len(replay) == 2


def test_aborts_excluded_corruption_and_transaction_rollback(tmp_path):
    with ReplayBuffer(tmp_path / "replay.sqlite3", 10) as replay:
        aborted = play_game(ScriptedSearch([]), game_id="aborted", seed=9, cancel=lambda: True)
        replay.add_game(aborted)
        assert len(replay) == 0 and replay.statistics()["aborted"] == 1
        # An insertion error must roll back the game ledger and all sample changes.
        replay.db.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON samples BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with pytest.raises(Exception, match="test failure"):
            replay.add_game(mate_record())
        assert replay.statistics()["completed"] == 0
        replay.db.execute("DROP TRIGGER fail_insert")
        replay.add_game(mate_record())
        replay.db.execute("UPDATE samples SET state = ?", (b"corrupt",))
        replay.db.commit()
        with pytest.raises(ValueError, match="Corrupted replay"):
            replay.sample(1, np.random.default_rng())


def tiny_config(workers):
    return Config(device="cpu", network_channels=8, residual_blocks=1, cpu_threads=1,
        batch_size=2, mcts_simulations=1, selfplay_workers=workers, games_per_generation=2,
        max_game_plies=2, replay_buffer_size=8)


def test_serial_and_spawn_workers_reproduce_and_shutdown(tmp_path):
    previous = {p.pid for p in mp.active_children()}
    serial = run_batch(tiny_config(1), tmp_path / "serial")
    parallel = run_batch(tiny_config(2), tmp_path / "parallel")
    assert {p.pid for p in mp.active_children()} == previous
    assert serial["replay"]["truncated"] == parallel["replay"]["truncated"] == 2
    assert serial["replay"]["draws"] == 0
    assert len({g["seed"] for g in parallel["games"]}) == 2
    for a, b in zip(serial["games"], parallel["games"]):
        first, second = load_game(Path(a["archive"])), load_game(Path(b["archive"]))
        assert first.seed == second.seed
        assert [s.action for s in first.samples] == [s.action for s in second.samples]
        for sa, sb in zip(first.samples, second.samples):
            np.testing.assert_array_equal(sa.policy, sb.policy)


def test_cancelled_batch_has_no_orphan_workers(tmp_path):
    previous = {p.pid for p in mp.active_children()}
    calls = [0]
    def cancel():
        calls[0] += 1
        return calls[0] >= 3  # Cancel after submitting the first wave.
    result = run_batch(tiny_config(2), tmp_path, cancel=cancel)
    assert result["cancelled"]
    assert result["replay"]["positions"] == 0
    assert {p.pid for p in mp.active_children()} == previous


def test_distinct_game_seed_streams():
    assert len({game_seed(42, i) for i in range(1000)}) == 1000
    with pytest.raises(ValueError):
        game_seed(42, -1)


def test_real_neural_mcts_completes_a_game(tmp_path):
    config = replace(tiny_config(1), games_per_generation=1, mcts_simulations=128, max_game_plies=8)
    report = run_batch(config, tmp_path, start_fen="7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    assert report["replay"]["completed"] == 1
    record = load_game(Path(report["games"][0]["archive"]))
    assert record.termination == "checkmate" and record.white_result == 1
    assert record.samples and all(s.policy_source == "mcts_visits" for s in record.samples)


def test_policy_only_archive_is_honestly_labeled(tmp_path):
    report = run_batch(replace(tiny_config(1), games_per_generation=1, mcts_enabled=False), tmp_path)
    record = load_game(Path(report["games"][0]["archive"]))
    assert all(s.policy_source == "network_policy" for s in record.samples)


def test_incompatible_archive_and_replay_schema_fail(tmp_path):
    import json
    path = tmp_path / "game.npz"
    save_game(mate_record(), path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(data["metadata"].item())
    metadata["data_schema"] = 999
    data["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="schema"):
        load_game(path)
    database = tmp_path / "replay.sqlite3"
    with ReplayBuffer(database, 4) as replay:
        replay.db.execute("UPDATE metadata SET value='999' WHERE key='replay_schema'")
        replay.db.commit()
    with pytest.raises(ValueError, match="schema"):
        ReplayBuffer(database, 4)
