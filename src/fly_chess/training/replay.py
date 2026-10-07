"""SQLite transactions: insert a game once and evict oldest positions atomically."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from fly_chess.core.actions import ACTION_SIZE
from fly_chess.core.encoding import INPUT_PLANES
from fly_chess.selfplay.records import SCHEMAS, POLICY_SOURCES, GameRecord, Sample


class ReplayBuffer:
    def __init__(self, path: Path, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity = capacity
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            with self.db:
                self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                existing = dict(self.db.execute("SELECT key, value FROM metadata"))
                expected = {**SCHEMAS, "replay_schema": 1}
                if existing and existing != {k: str(v) for k, v in expected.items()}:
                    raise ValueError("Incompatible replay schema")
                self.db.executemany("INSERT OR IGNORE INTO metadata VALUES (?, ?)", [(k, str(v)) for k, v in expected.items()])
                self.db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, digest TEXT NOT NULL, metadata TEXT NOT NULL)")
                self.db.execute("""CREATE TABLE IF NOT EXISTS samples (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT NOT NULL, ply INTEGER NOT NULL,
                    state BLOB NOT NULL, legal BLOB NOT NULL, policy BLOB NOT NULL, info TEXT NOT NULL,
                    UNIQUE(game_id, ply))""")
                self._prune()
        except Exception:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def _prune(self):
        excess = len(self) - self.capacity
        if excess > 0:
            self.db.execute("DELETE FROM samples WHERE seq IN (SELECT seq FROM samples ORDER BY seq LIMIT ?)", (excess,))

    def add_game(self, record: GameRecord) -> bool:
        record.validate()
        metadata = json.dumps(record.metadata(), sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(metadata.encode())
        rows = []
        for ply, sample in enumerate(record.samples):
            info = json.dumps({"action": sample.action, "white_to_move": sample.white_to_move,
                "value_prediction": sample.value_prediction, "policy_source": sample.policy_source, "target": sample.target}, sort_keys=True)
            state = zlib.compress(sample.state.astype("<f4").tobytes())
            legal = sample.legal_actions.astype("<u2").tobytes()
            policy = sample.policy.astype("<f4").tobytes()
            for part in (info.encode(), state, legal, policy):
                digest.update(part)
            rows.append((record.game_id, ply, state, legal, policy, info))
        checksum = digest.hexdigest()
        with self.db:
            old = self.db.execute("SELECT digest FROM games WHERE id=?", (record.game_id,)).fetchone()
            if old:
                if old[0] != checksum:
                    raise ValueError("Game ID already exists with different data")
                return False
            self.db.execute("INSERT INTO games VALUES (?, ?, ?)", (record.game_id, checksum, metadata))
            # Cancelled games are archived/countable but never become training targets.
            if record.status != "aborted":
                self.db.executemany("INSERT INTO samples (game_id, ply, state, legal, policy, info) VALUES (?, ?, ?, ?, ?, ?)", rows)
            self._prune()
        return True

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[dict]:
        if type(batch_size) is not int or not 1 <= batch_size <= len(self):
            raise ValueError("batch_size must be positive and no greater than current replay size")
        ids = np.asarray([row[0] for row in self.db.execute("SELECT seq FROM samples ORDER BY seq")], dtype=np.int64)
        output = []
        for seq in rng.choice(ids, size=batch_size, replace=False):
            row = self.db.execute("SELECT game_id, ply, state, legal, policy, info FROM samples WHERE seq=?", (int(seq),)).fetchone()
            try:
                game_id, ply, state, legal, policy, info = row
                fields = json.loads(info)
                sample = Sample(np.frombuffer(zlib.decompress(state), dtype="<f4").reshape(INPUT_PLANES, 8, 8).copy(),
                    legal_actions=np.frombuffer(legal, dtype="<u2").copy(), policy=np.frombuffer(policy, dtype="<f4").copy(), **fields)
                if (not np.isfinite(sample.state).all() or sample.policy.shape != sample.legal_actions.shape
                        or not len(sample.policy) or not np.isfinite(sample.policy).all() or (sample.policy < 0).any()
                        or not np.isclose(sample.policy.sum(), 1, atol=1e-6)
                        or (sample.legal_actions >= ACTION_SIZE).any() or sample.action not in sample.legal_actions
                        or len(np.unique(sample.legal_actions)) != len(sample.legal_actions)
                        or not np.isfinite(sample.value_prediction) or abs(sample.value_prediction) > 1
                        or sample.policy_source not in POLICY_SOURCES or type(sample.white_to_move) is not bool
                        or sample.target not in (-1., 0., 1.)):
                    raise ValueError("Invalid stored replay sample")
                output.append({"game_id": game_id, "ply": ply, "sample": sample})
            except Exception as exc:
                raise ValueError(f"Corrupted replay sample {seq}: {exc}") from exc
        return output

    def statistics(self) -> dict:
        stats = {key: 0 for key in ("completed", "truncated", "aborted", "white_wins", "black_wins", "draws")}
        for (raw,) in self.db.execute("SELECT metadata FROM games"):
            meta = json.loads(raw)
            stats[meta["status"]] += 1
            if meta["status"] == "completed":
                stats[{1: "white_wins", -1: "black_wins", 0: "draws"}[meta["white_result"]]] += 1
        return {**stats, "positions": len(self), "capacity": self.capacity}
