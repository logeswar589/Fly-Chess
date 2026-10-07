# Self-play and replay contracts — Phase 4

## Game generation

`play_game(search, game_id=..., seed=..., start=..., max_plies=...)` uses the same frozen neural model for White and Black. Every move uses MCTS with self-play root noise (or an explicitly configured policy-only experiment). The default temperature is 1 for the first 30 generated plies, then 0. Recorded policies are the untempered visit targets, independent of move sampling.

Each sample stores a float32 encoded state, selected action, ordered legal action IDs, probabilities aligned with those IDs, player color, raw network value prediction, target, and policy provenance. Legal IDs are a sparse mask: reconstruct a full mask by setting those action indices true. Some legal actions legitimately have zero visit probability.

A game record stores root FEN plus prefix move history, model/game IDs, game seed, search/generation settings, schema versions, status, termination reason, and White-relative result. Root FEN and all moves preserve repetition information. Validation reconstructs the game and checks state encodings, every legal-action list, policy normalization, turn perspective, move legality, final rules outcome, and labels.

Statuses are distinct:

- `completed`: the rules produce a win or draw. The final result is converted separately to each stored player-to-move perspective.
- `truncated`: the configured generated-ply limit is reached. Target zero is an explicit training convention, not a claimed chess draw. These samples are included in replay and are distinguishable through their game metadata.
- `aborted`: cancellation or interrupted search before a game result. Partial samples have no outcome target and are excluded from replay positions. The game is still archived and counted.

Search exceptions propagate as failures; they are not silently turned into draws. An MCTS fallback policy or policy-only experiment retains its own provenance label. There is no optimizer step or claim of model improvement in this phase.

## Archives and replay

Each game is a compressed `.npz` under `data/selfplay`. Numerical arrays contain states and sparse policy/action vectors with offsets; a JSON string holds metadata. Loading disables pickle, checks schema/array structure, and then validates the complete reconstructed game. Saving writes and flushes a temporary file in the same directory before atomic replacement.

`ReplayBuffer(path, capacity)` is a SQLite store under `data/replay_buffer`. Each sample stores compressed float32 planes, uint16 legal IDs, float32 probabilities, and metadata. One transaction inserts the game ledger entry and its samples, then evicts the oldest excess positions. Capacity is measured in positions, not games. A partially retained old game is valid because each sample already contains its final target.

Adding an identical game ID/content again is a no-op, including after its samples have been evicted. Reusing an ID with changed content is an error. Game ledger entries remain for deduplication and lifetime outcome counts. Sampling uses a supplied NumPy generator, stable sequence order, and selection without replacement. Load/sample errors are explicit; corruption is not silently skipped. Reopening enforces the requested capacity and validates the schema.

Archive retention and the deduplication ledger grow with generated games; the position cap does not cap total disk storage. SQLite can retain previously allocated pages. Automated archive retention/compaction is not implemented. Archives are written before replay commits so an archive remains available if a subsequent replay write fails. Recovery can reload an archive and call `add_game`; automatic reconciliation belongs to the Phase 5 pipeline.

## Processes and reproducibility

`run_batch` accepts Config, workspace, optional inference weights, starting FEN, cancellation callback, starting game index, and a committed-game progress callback. One-worker mode runs inline. Multiple workers use an explicit `spawn` process pool; each child initializes its own model/device and RNG. Only the parent writes archives/replay.

Games use `seed = (base_seed + game_index * 0x9E3779B9) mod 2**32`, a distinct deterministic stream for each uint32 index. Serial and parallel scheduling share the same game-index mapping. The model initialization seed stays fixed so all workers use the same initial network. When weights are supplied, keep the file immutable during the batch. Future generations must advance `start_index`; the standalone command starts at index zero and therefore deliberately reproduces the same seeded games on repeated runs, under new archive IDs.

Submission happens in bounded waves, at most one game per worker per wave. The parent commits results in input order, making replay eviction independent of completion order. A shared event propagates cancellation, including inside MCTS; the parent polls it while waiting. `finally` sets cancellation and joins the pool on completion, error, or keyboard interruption. An in-flight inference is cooperative and can delay shutdown. An OS kill is outside this graceful-shutdown guarantee.

Call multiprocessing code from a guarded executable entry point (`if __name__ == '__main__':`). The supplied module and console entry points support Windows spawning. CPU threads are limited per worker. For CUDA, each worker has its own model/context; start with one worker and measure before increasing counts.

## Verified behavior

Tests cover winning targets for both colors, actual draws, truncations, aborted exclusion, complete-game archive round trips, schema/corruption failures, transactional rollback, capacity pruning, conflict detection, seeded sampling after restart, serial/two-process equivalence, cancellation shutdown, and a completed real neural-MCTS game from a near-mate fixture. The CLI smoke preset intentionally truncates two short games; its records are not evidence of chess strength.
