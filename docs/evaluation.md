# Model competition and statistics — Phase 6

## Match protocol

After each generation, the controller freezes a full candidate checkpoint and evaluates it against `models/fly_best.pt`. If no best exists, the first candidate becomes an explicitly labeled initial baseline, with no games, rating, or improvement claim. Later candidates require a successful evaluation gate to replace it.

The default match has 50 opening pairs / 100 games. Each pair starts from the same legal opening prefix, with candidate colors swapped. Prefixes are sampled independently with replacement from a small fixed distribution listed in each report. This measures performance under that distribution, not universal chess strength. Each player has the same MCTS simulation/depth/node/PUCT budgets. Evaluation uses temperature zero, no self-play noise, and no opponent adaptation. Seeds, starting positions, moves, termination reasons, checkpoint identities/hashes, search settings, and runtime version are persisted.

A ply-limit ending is `truncated`, never a claimed rules draw. Any truncation or incomplete pair makes the match ineligible for promotion. Completed wins, losses, and true draws are counted separately. Evaluation games do not enter training replay.

## Conservative promotion rule

For each color-swapped pair, average its two candidate scores (win=1, draw=0.5, loss=0). Treat that pair as one bounded observation in [0,1]. With n complete independently sampled pairs, use the two-sided Hoeffding interval:

`mean ± sqrt(log(2 / alpha) / (2n))`, clipped to [0,1].

Default alpha is 0.05. A candidate is promoted only after the entire requested match finishes without truncation, at least `evaluation_min_pairs` complete pairs exist (default 20), and the interval's lower bound exceeds 0.5. No early stopping for a favorable interim score. Pairing avoids falsely treating color-swapped games as independent observations. The bound is deliberately conservative and may retain the incumbent despite a positive raw score.

Synthetic test records verify this rule; production results come from actual matches. The 100-game default does not guarantee statistical significance, improving strength, or best-model promotion.

## Honest ratings and progress

Report candidate win rate and score rate separately: draws contribute half a point to score but are not wins. For an eligible match with score strictly between zero and one, `400 * log10(score / (1-score))` is reported as a **relative Elo difference versus that particular incumbent**, conditional on the match setup. The score interval is transformed where finite. At 0% or 100%, no finite point estimate is invented; bounds that extend to infinity are null/unbounded.

Small, incomplete, or truncated evaluations are Unrated. No calibrated absolute or human/federation Elo exists, so training `estimated_elo` remains None. Relative differences against changing opponents must not be connected as if they were a common absolute rating scale. Future charts can display score/win rates with opponent/sample context; a continuous absolute Elo curve requires a stable calibrated anchor not implemented here.

Training checkpoints persist actual generation losses, learning rates, self-play outcomes, average generated game length, and evaluation summaries. Self-play White/Black counts are not candidate-versus-incumbent wins/losses. Missing evaluations stay missing; metrics are never interpolated into fictitious measurements.

## Journals, interruption, and promotion

`logs/evaluation/<id>.json` is atomically written after each finished game. Resume with the same candidate, settings, and evaluation ID to continue from the next game in the same pair sequence. An aborted game's partial moves are retained for inspection but the game restarts on resume. Candidate/setting mismatches and changed incumbents fail explicitly. A separate evaluation lock prevents simultaneous promotions in a workspace.

The final report is saved before atomically replacing best. Best contains its promotion report and evaluation ID. A failure before replacement leaves the incumbent intact; rerunning the same ID uses completed journal games to retry. A failure after replacement is detected by the embedded ID and does not rerun/promote again. Reports identify the initial baseline separately from competitive promotion.

Automatically generated candidate/best checkpoints retain model, optimizer, scheduler, RNG, replay manifest, and progress, so they can be used for training resume within the preserved workspace. Standalone evaluation also accepts inference-only weights; if the source lacks training state, its promoted copy remains inference-only. Latest continues training independently of best; a rejected candidate does not overwrite best or silently roll back the optimizer.

## Configuration and commands

- `evaluation_enabled`: default true; false explicitly skips the gate and creates no best model.
- `evaluation_pairs`, `evaluation_min_pairs`, `evaluation_simulations`, `evaluation_max_plies`: configurable budgets.
- A smoke preset uses very few short games and is intentionally ineligible for meaningful promotion.
- Phase 5 checkpoints without these settings receive the Phase 6 defaults on resume; inspect budgets before starting a long evaluation. Previously saved fields remain authoritative.

Standalone example: `python -m fly_chess --config configs/smoke.toml evaluate --candidate models/fly_latest.pt --id trial-001`. The controller performs evaluation automatically between optimization and generation completion. Pause/stop can interrupt evaluation between search operations; the training checkpoint and journal retain its position. No external chess engine or rating service is used.
