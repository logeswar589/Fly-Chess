# Fly-Chess implementation status

- Current state: Local release 0.1.0. All ten implementation phases are complete, with source/wheel packaging and integrated workflow verification. No playing-strength claim.
- Completed phases: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
- Remaining research: sustained strength training/evaluation and long-run resource characterization; neither is required to use the delivered local release.
- User direction: continue phase-wise without asking permission at each phase. No clarification pending.
- Specifications: PHASE_PLAN.md; docs/contracts.md, neural.md, search.md, selfplay.md, training.md, evaluation.md, ui.md, training-ui.md, human-learning.md.

## Entry points

Launch from this workspace:

```powershell
.\.venv\Scripts\python.exe -m fly_chess gui
.\.venv\Scripts\python.exe -m fly_chess gui --weights logs/evaluation-smoke/models/fly_best.pt
```

The second command uses a smoke checkpoint, not a strength-trained model. The main models directory has no strength-trained weights. GUI setup has an honest untrained path and accepts dropped .pt files. Global --workspace and --config precede the subcommand.

## Implemented

- Phases 1–4: strict configuration, python-chess rules/encoding, residual policy/value network, legal CPU/CUDA inference, PUCT MCTS, neural self-play, compressed archives, transactional replay, Windows-safe process workers.
- Phases 5–6: actual optimization, durable optimizer/scheduler/RNG/replay checkpoints, generation state machine, paired candidate evaluation, conservative promotion, locking and interrupted-run recovery. Absolute Elo remains unknown.
- Phase 7: Pygame CE monochrome UI, bundled licensed fonts, original vector pieces, White/Black play, promotion, legal marks, SAN, undo, claim draw, resignation, resizing, keyboard controls, motion toggle. One inference worker rejects stale game/revision tokens. Brain tabs expose actual module/input/activation/parameter samples, legal policy, MCTS visits/Q/tree and recorded decisions.
- Temporary opponent adaptation is off by default, session-only, bounded below 20%, regularized and conditioned on legal opportunities. It scores before observing, updates only human choices, reconstructs on undo and never writes main weights. No demonstrated strength benefit.
- Phase 8: spawned-process Start/Resume/Pause/Stop/Save/Load controls. Actual replay-position before/after captures, target/predicted policy/value, gradient/weight/update norms, selected weight histograms, configurable diagnostics. Watch replays archived games at 0.25x/0.5x/1x/2x/MAX with exact archived-model inference. Charts use persisted metrics and explicit missing values.
- Phase 9: consent-gated asynchronous PGN storage on New/Exit, incomplete and resigned semantics, separate immutable human datasets, explicit supervised candidate training and the existing evaluation gate. No human data enters self-play replay. Save failures retain their board snapshots for retry; closing cancels/rejects late AI moves before archival.

## Verification — 2026-10-07

Windows / Python 3.14.5 / chess 1.11.2 / NumPy 2.5.3 / pytest 9.1.1 / Pygame CE 2.5.8 / PyTorch 2.11.0+cu128. NVIDIA RTX 3050 6GB Laptop GPU, driver 616.92.

- Phase 10 full regression: 139 passed in 52.86s. Every module was reached through the real inspector worker; input and weight samples matched exactly and hooks were cleaned up. New-match checkpoint refresh and same-match weight freezing are tested.
- Integrated source workflow: logs/release-25aae178. Train -> play White -> close -> new app/load -> resume -> paired evaluation -> play Black with step-4 model replacing step-2. Four training games / four optimizer steps / four evaluation games, no unjustified promotion, no surviving registered child processes.
- Fresh wheel target workflow: logs/package-check-7f9da0fd/verification.json. Package imports resolved to the isolated installed wheel directory outside the project working directory. Same workflow passed with no orphaned child processes. Existing dependency installation was reused; this is not a clean-OS installation claim.
- Release artifacts: dist/fly_chess-0.1.0-py3-none-any.whl, dist/fly-chess-0.1.0-source.zip and dist/manifest.json. Source launcher: Start-Fly.ps1. Full verification/recovery/installation notes: docs/release.md.
- Matched scripted adaptation diagnostic: logs/adaptation-experiment.json, four on/off pairs, two styles, both colors, eight simulations and 32-ply cap. All eight games truncated, unchanged main weights, no established strength benefit; remains experimental/off by default.
- Visual review found and fixed a training timestamp/title overlap. Menu/play/training/brain/config/error renders checked at reference, smaller and larger sizes. Native OS interaction/high-DPI configuration variations were not automated.

- Phase 9 full regression: 136 passed in 53.79s. One third-party python-chess asyncio deprecation warning on Python 3.14.
- After the final close/archive race guard, the relevant human UI/data tests passed: 19 in 6.97s, including the new late-AI/close regression. Compileall and dependency checks passed.
- Existing suite includes CPU exact interrupted/uninterrupted training, fresh-process restore, CUDA optimizer/RNG restore, failed checkpoint/optimizer commits, replay reconciliation, locks, multiprocess self-play, paired evaluation/promotion recovery and actual chess rules.
- New tests cover stale play replies, both human colors, special chess rules, adaptation reset/undo/forced moves, immutable weights, progress telemetry equivalence, bounded worker messages, promotion UI, freeze retention, training process pause/save/exit/reload/resume, watch speeds, missing chart data, independent gradient capture, PGN consent/round-trip/results, human candidate isolation/gate rejection, save failure/retry, and exit-time snapshot consistency.
- Real GUI training smoke: logs/gui-training-a2f4b2d2 completed two truncated eight-ply games / two optimizer updates; 309 render/input frames, mean frame work 3.93ms, max 29.49ms. Screenshots under its screenshots folder include learning, stats, watch and config at 1440x960 and 1200x800.
- Real GUI human-learning smoke: logs/human-ui-8d56538e saved a legal complete fixture PGN, learned from two human positions in one update, rendered 266 frames, evaluated and retained the incumbent. Candidate and exact dataset provenance are preserved there.
- Reviewed menu/play/layers/search/settings screenshots under logs/ui-preview, training/Watch screenshots and the human-learning screen. These are real Pygame renders, using the offscreen SDL driver rather than a native desktop interaction test.
- CUDA microbenchmark: logs/telemetry-benchmark.json. Five warm trials: 32-simulation search ~153–159ms with/without capture (noise); update batch 8 ~24.45ms without vs 41.61ms with diagnostics every step. GUI defaults to sampling every 8 updates. This is kernel/instrumentation work, not full training throughput.

Use a fresh UUID --basetemp under .pytest_cache for pytest; the system shared temporary directory has a Windows permission issue. README gives the exact command. scripts/render_ui.py, verify_training_ui.py, verify_human_ui.py and benchmark_telemetry.py reproduce visuals/smokes. Their outputs are confined to logs; no long strength-training run was performed.

## Maintenance contracts

- TrainingController stages: selfplay -> optimize -> evaluate -> finish. Partial latest checkpoints retain the next operation. Failed in-memory controllers must reload. Resume config must match saved state. Keep replay snapshot/data with full checkpoints.
- Training runs in a spawned process; inference models stay in their own threads. Do not share mutable networks or global training RNG with the GUI. UI inference uses frozen checkpoint snapshots. Model choice in setup affects the next match, not the existing match.
- Human archives require explicit saving opt-in (off at launch). New/Exit saves the surviving history; incomplete PGNs stay excluded. Human candidates are inference-only and do not overwrite full latest self-play checkpoints. A promoted human best cannot serve as a full self-play resume file.
- Promotion requires all requested pairs complete, minimum sample count, and the paired 95% Hoeffding lower score bound above 0.5. Defaults are 50 opening pairs / 100 games, 20 minimum complete pairs. Baseline creation is labeled; human candidates first establish the unchanged parent as baseline if necessary, then still compete.
- Stop/pause/close finish a self-play wave or minibatch; evaluation stops between searches. No forced shutdown. Keep rendering while waiting. Slow viewers may drop display reports, never durable training events.
- All telemetry is bounded and labeled by source. Never imply old/frozen samples describe the current board. Missing activation/archive/evaluation/rating data must remain explicit. Large activation captures are transient, not accumulated in checkpoints.
- Checkpoint/PGN/dataset/metric retention is not pruned automatically. Long-run storage/memory behavior and measured playing strength remain uncharacterized. Package isolation/asset/workflow checks passed. No calibrated Elo or improvement claim exists.

## Brain view follow-up — 2026-10-07

- Added default Brain tabs to Play and Train: fly-inspired 3D activation clouds, scroll zoom, drag rotation, exact-value hover, reset and freeze. Training compares real before/after captures with shared layer scales.
- Separate bounded telemetry samples every leaf module (48 values/output, 8,192 maximum); coordinates are illustrative and do not claim anatomical or synaptic fidelity. Values refresh on measured forwards/optimizer captures, not decorative animation. Self-play/evaluation retain the latest optimizer capture.
- Full regression: 146 tests passed. Real training/render workflow: logs/gui-training-72aefffe, 2 games/2 updates, 276 frames, mean frame work 6.46 ms. Screenshot review at 1200×800 and 1440×960.
