# Fly-Chess — phased implementation plan

## Objective and scope

Build a modular Python/PyTorch chess agent that learns from neural-network self-play, saves and resumes training, and plays humans in a responsive Pygame interface. Use python-chess for rules. No chess-engine wrapper, fabricated metrics, or promised strength after a specific number of generations.

This document is the implementation specification. Build one phase per request. A phase is complete only when its acceptance checks pass or any hardware-dependent checks are explicitly recorded as unverified. Do not call the whole application complete before Phase 10.

## Token-efficient working process

1. At the beginning of a phase, read this phase, `STATUS.md`, and relevant source files. Do not repeatedly paste the original brief or scan the entire repository.
2. Implement only the active phase and necessary fixes to earlier work. Ask only about decisions that materially change scope; use the defaults below otherwise.
3. Run focused checks. Preserve their results in `STATUS.md`; do not include large logs in chat.
4. End with a short summary: changed behavior, verification, remaining limitations, and next phase.
5. Update `STATUS.md` before stopping so another chat can continue without reconstructing history.
6. If a phase is too large, split it at a working boundary and record the remaining work. Never trade correctness for an arbitrary token cap.

Reusable request:

> Implement Phase N from PHASE_PLAN.md. Read STATUS.md first. Inspect only relevant files, preserve the documented contracts, complete the phase's acceptance checks, and update STATUS.md. Stop at the phase boundary and report results concisely. Do not start the next phase or a long training run.

## Fixed contracts established in Phase 1

- Separate modules for rules/encoding, network, search, self-play, replay storage, optimization, checkpoints, evaluation, human games, and UI.
- Use an absolute board orientation: a1 through h8 use python-chess square ordering. Document tensor row/column mapping and test it.
- Use a stable 20,480-action space: `(from_square * 64 + to_square) * 5 + promotion_code`, with promotion codes none, knight, bishop, rook, queen. Unused actions are always masked. This intentionally favors a simple, testable mapping; changing it requires a checkpoint/data schema migration.
- Position planes include 12 piece planes, side to move, four castling-right planes, en-passant square, halfmove clock, and repetition indicators. Finalize exact order, scaling, and version before creating training data. Retain full rules/history information in the game state for search and draw adjudication.
- Network value and training value targets always use the player-to-move perspective. Convert the final White-relative result for each stored position; never mix perspectives.
- Network returns policy logits and a bounded scalar value. Apply legal masking before probabilities or move selection. Terminal positions return an outcome without selecting a move.
- Self-play stores encoded state, legal action IDs/mask, selected action, MCTS policy, value prediction, final target, game ID, termination reason, and schema version. Store sparse policy/mask data where practical.
- MCTS uses PUCT; store edge values from the parent player's perspective and explicitly convert signs during backup. Search must retain history relevant to repetition.
- Truncation by a configured ply limit is recorded separately from a rules-based draw. If assigned target zero, disclose that training convention in metrics.
- CPU is supported first; `device=auto` selects usable CUDA when available. Keep device selection explicit and visible.
- UI owns rendering and input. Training/search run outside its event loop. Worker messages are structured; stale search results carry game/request IDs and are rejected after undo, reset, or model changes.
- Checkpoints and replay schemas are versioned. Write atomically, validate before activation, and preserve the last valid save on failure.

Suggested structure:

```text
src/fly_chess/
  config.py
  core/          # rules, encoding, action mapping
  neural/        # policy-value network and inference
  search/        # MCTS
  selfplay/      # game generation and worker orchestration
  training/      # replay, losses, optimizer, pipeline
  storage/       # checkpoints, datasets, PGN, metrics
  evaluation/    # candidate-versus-best arena
  game/          # human game controller
  ui/            # menu, board, training, statistics, settings
configs/         # lightweight and larger presets
tests/
data/selfplay/
data/human_games/
data/replay_buffer/
models/
logs/
```

## Phase 1 — Foundation, chess rules, and encoding

Deliver project packaging, CLI entry point, configuration validation, logging, directory creation, and the core chess adapter. Implement action encoding/decoding, board planes, legal masks, and outcome handling. Document the fixed contracts above. Add a lightweight CPU configuration with conservative, configurable search and replay budgets; record its measured performance later rather than inventing throughput.

Acceptance:
- Fresh environment can install the project and run a headless diagnostic.
- Every legal move round-trips through action IDs in representative positions, including every promotion, castling, and en passant.
- Encoding, turn changes, checkmate, stalemate, repetition, and draw-claim policy have focused tests.
- Invalid configuration fails with an actionable message.

## Phase 2 — Trainable neural network and inference

Implement a compact residual convolutional network with policy/value heads, batched inference, device selection, deterministic seed controls, and legal probability masking. Add a policy-only agent for diagnostics, clearly labeled untrained initially.

Acceptance:
- Forward outputs have documented shapes, finite values, and bounded value predictions.
- Legal probabilities sum to one; illegal actions receive zero probability.
- Terminal states and numerical failures have explicit handling.
- A synthetic optimizer step updates parameters with finite gradients. Weight save/load preserves inference within tolerance.

## Phase 3 — MCTS search

Implement PUCT selection, network expansion, terminal evaluation, sign-correct backup, and visit-count policies. Add self-play-only root noise, temperature controls, simulation/time limits, and cancellation. Support a documented policy-only experimental mode; its targets must not be misrepresented as MCTS-improved policies.

Acceptance:
- Constructed search fixtures verify value signs and visit accounting.
- Forced mate/terminal positions and all returned actions obey chess rules.
- Simulation limits, cancellation, and repetition history behave correctly.
- Fixed-seed search is reproducible on the tested device within documented limits.

## Phase 4 — Self-play and replay storage

Generate complete neural-agent games through MCTS. Initially use one worker; introduce configurable Windows-safe process workers after the serial path works. Persist game records and a bounded replay buffer; convert outcomes to each position's player-to-move target. Track completed, aborted, and truncated games separately.

Acceptance:
- A small seeded self-play batch runs, reloads, and yields valid training samples.
- Samples contain normalized legal policies and correct terminal targets for both colors.
- Replay capacity, restart loading, process shutdown, and malformed-data handling work.
- Single-worker and multi-worker runs use distinct reproducible seed streams and leave no orphan workers.

## Phase 5 — Training and durable resume

Implement minibatch policy cross-entropy against search targets, value error, regularization, optimizer/scheduler, gradient controls, and the generation pipeline. Support start, pause, stop, save, and resume through a headless controller before connecting the GUI.

Checkpoint contents: model, optimizer, scheduler, configuration, schema/network versions, generation and current pipeline stage, completed games, training steps, statistics, RNG states, replay manifest, and evaluation metadata. Capture CUDA RNG when applicable. Save generation files plus `fly_latest.pt`; reserve `fly_best.pt` for evaluated promotion.

Define recovery precisely: resume at the last durable game/minibatch boundary inside the recorded generation. Persist completed work before advancing its progress marker. An in-progress game may restart after a crash; never claim exact mid-search recovery. Manual pause/save must reach a consistent boundary and acknowledge completion.

Acceptance:
- A tiny real self-play → update → save → fresh-process load → update cycle succeeds.
- Resume restores optimizer/scheduler, counters, replay references, and RNG state, not only weights.
- Interruption tests cover a partial generation and interrupted save without corrupting the previous checkpoint or duplicating completed work.
- Missing, incompatible, or corrupted checkpoints produce useful errors.

## Phase 6 — Model competition and truthful statistics

Evaluate frozen candidate and best checkpoints with balanced colors, matched starting positions, fixed search budgets, and recorded seeds. Configure evaluation size; 100 games is an example, not a statistical guarantee. Establish a documented promotion rule with a confidence interval suitable for paired games. Keep the incumbent when evidence is insufficient. Bootstrap the first best model with an explicit baseline label.

Separate self-play White/Black/draw counts from candidate-versus-opponent wins/losses/draws. An unanchored rating is a relative internal estimate, never a human/federation Elo. Show `Unrated` when data are insufficient; include opponent, sample size, uncertainty, and evaluation settings when displaying estimates.

Acceptance:
- Known evaluation records produce correct scores and promotion decisions.
- Draws, small samples, interrupted evaluation, and extreme scores are handled without invented or infinite displayed ratings.
- Only a successful promotion changes `fly_best.pt`.
- Metrics are persisted from actual events, including losses and average game length.

## Phase 7 — Human-versus-Fly graphical game

Build the Pygame menu and board: click-to-move, legal highlights, promotion chooser, board flip, move history, captured-piece display, turn/status, thinking indicator, animation, and game completion. Add model selection, White/Black/random side, new game, resign, undo, and main menu. Run AI search asynchronously.

Map Easy/Medium/Hard/Full Power to documented search budgets and controlled sampling; enforce a resource cap even at Full Power. No trained checkpoint should show an honest empty state with a route to training. Optional evaluation and clock settings may be added here; define time-control and timeout rules before exposing a competitive clock.

Acceptance:
- Complete human games against saved checkpoints from either side.
- Promotion, castling, en passant, mate, draw, resignation, undo, and board orientation behave correctly.
- Undo during search cancels/rejects stale results and restores the intended human turn.
- Input/rendering remain responsive during search; loading errors are visible.

## Phase 8 — Training UI, Watch Fly, and graphs

Connect the tested controller to Train Fly with start/pause/stop/save/load. Display generation, actual game counts, clearly labeled outcomes, learning rate, game length, losses, rating status, and active training time. Add Training Stats and Settings screens.

Watch Fly replays or subscribes to game events, with 0.25x/0.5x/1x/2x/MAX speed. Rendering must not block training; MAX suppresses most visual updates. Bound event queues so a slow viewer cannot exhaust memory. Implement graphs for games versus rating, generation versus evaluation win rate, all three losses, and average game length. Represent missing evaluations as missing data.

Acceptance:
- Train → pause → save → exit → reopen → resume works from the GUI.
- Charts match persisted metrics and handle empty history.
- Watch speeds work without changing stored training results or blocking workers.
- Model/config changes during active work follow explicit controller state rules.

## Phase 9 — Optional learning from human games

Add opt-in human-game PGN storage with positions, actions, colors, results, checkpoint identity, and provenance. Add an explicit Train From Human Games action using a separate dataset and documented supervised objective. Decide how incomplete, resigned, and undone games are recorded; do not silently turn incomplete records into outcome targets.

Acceptance:
- Saved complete PGNs round-trip through python-chess.
- Storage respects opt-in; human moves never silently enter self-play replay.
- Explicit human-data training creates a candidate checkpoint and uses the same evaluation gate before becoming best.

## Phase 10 — Integration, performance, and release

Run the complete lightweight CPU workflow, tune defaults from measurements, document installation/launch/training/resume, and prepare the runnable project with assets and dependency definitions. Test CUDA only if available and report its status honestly. Document checkpoint trust, storage limits, recovery boundaries, and realistic training cost.

Acceptance:
- Fresh setup launches all menu modes.
- End-to-end: train → save → play → close → load → train more → compare → play.
- No illegal moves, UI freezes from synchronous compute, orphan workers, silent data loss, fake ratings, or automatic best-model replacement outside the evaluation gate in tested scenarios.
- Tests cover the critical contracts and failure paths; visual review checks board/UI layout at supported sizes.
- README includes measured machine/device details, known limitations, and exact launch commands.
- Deliver working software separately from any claim of chess strength. Meaningful strength requires additional measured training and evaluation.

## Release milestones

- Phases 1–3: legal neural agent with search.
- Phases 4–6: headless train/save/resume/evaluate system.
- Phase 7: first playable graphical application.
- Phase 8: full training and observation interface.
- Phases 9–10: optional human learning and verified release.

## Scope discipline

Defer distributed training, cloud services, opening databases, external engines, large networks, and custom CUDA kernels. Optimize only measured bottlenecks. Keep the small architecture configurable so stronger hardware can later use larger budgets without rewriting the application.


## Required visual direction — bespoke monochrome UI

This is a core requirement, not an optional release polish task. Use a deliberate black-and-white chess study / technical instrument aesthetic. Avoid generic dashboard templates, repeated rounded statistic cards, gradients, glass effects, neon, decorative network backgrounds, oversized hero text, and excessive empty space.

- Palette: near-black, white, and a small set of neutral grays. Communicate state through labels, line patterns, shapes, and emphasis; never rely only on gray differences.
- Typography: one carefully chosen readable interface family plus tabular numerals or a restrained monospace face for telemetry. Bundle fonts with appropriate licenses.
- Layout: the chessboard is the primary focus during play. A structured side panel contains move history and brain inspection; secondary controls remain compact. Training centers the current game, meaningful progress, and the brain view. Use thin rules, consistent spacing, sharp geometry, and sparse borders.
- Pieces: clear, consistent chess silhouettes that remain distinct on both square colors. Highlights, selection, last move, and check use separate monochrome marks that do not obscure pieces.
- Interaction: coherent hover, pressed, selected, disabled, focus, loading, empty, and error states. Include keyboard navigation for core controls and legible scalable text.
- Motion: short functional transitions and move animation. Offer reduced motion. No continuous decorative animation.
- Responsive desktop behavior: support a documented minimum window size, resizable layouts, and high-DPI scaling. Brain inspection can expand without shrinking the board into an unusable thumbnail.

Before implementing the full GUI, Phase 7 begins with a UI design pass: establish tokens and build reviewable menu, play, and training-screen mockups in the project's UI technology. Capture screenshots and check hierarchy, readability, spacing, board contrast, and interaction states. Save the visual specification so later phases remain consistent. These mockups must not present invented training numbers as live measurements.

## Required Fly Brain visualization — playing and training

Visualize the actual implemented agent. "Complete brain" means coverage of every network module plus the search and learning pipeline, with drill-down into details. A simultaneous drawing of every scalar connection would be unreadable; provide an architecture overview and inspectable layer/channel/weight views instead. This is a chess neural network, not a biological fly connectome or a claim to reveal human-like thoughts.

The UI must identify the exact model/checkpoint, position, side to move, and telemetry timestamp. Distinguish live, sampled, frozen, and unavailable data. Keep every overlay aligned to its source position and discard stale updates.

### Playing: inference and decision view

- Overview: encoded board → convolutional/residual modules → policy/value heads → MCTS → chosen move. Include every actual module, with expandable residual blocks.
- Layer inspector: input planes, activation heatmaps, channel selection, activation distributions, tensor shapes, and parameter counts. Allow weight/filter inspection on demand. Label normalization scales so separate heatmaps are not misleadingly comparable.
- Policy inspector: legal moves ranked by raw network probability and MCTS visit share, with the selected move marked. Board overlays connect inspected moves to source/destination squares.
- Value inspector: show the predicted result from the labeled player-to-move perspective. Do not describe value as centipawns or a calibrated win probability without justification.
- Search inspector: show explored nodes, simulations, candidate visits, priors, value statistics, and the leading line. Use a bounded expandable tree and summary counts rather than rendering every node at once.
- Freeze and step: inspect the recorded root-position inference for a move. Clearly distinguish it from sampled evaluations of other positions inside MCTS.

### Training: learning view

- Display a sampled self-play position and its real forward-pass activations using the same inspectors as play.
- Show replay batch provenance, target policy versus predicted policy, target outcome versus predicted value, policy/value/total loss, learning rate, and optimizer step.
- Add per-layer gradient norms, weight norms, and update magnitudes at a configurable sampling interval. Offer weight histograms and selected filter inspection on demand.
- Show data flow through self-play, replay, minibatch optimization, checkpoint save, and candidate evaluation with actual stage status.
- Make it possible to freeze a sample and compare its outputs before and after an update on the same position. Do not imply every update improves chess strength.
- In ordinary human play, label the model as inference-only: no gradient updates occur unless explicit training is running.

### Implementation split and acceptance checks

- Phase 2: provide opt-in instrumentation interfaces and a stable module inventory. Telemetry is detached from autograd and does not retain computation graphs.
- Phase 3: expose compact search snapshots with root position/request identity, visits, priors, and value perspective.
- Phase 5: expose sampled learning metrics and before/after diagnostics without changing the optimization objective.
- Phase 7: deliver the monochrome design system and working inference/search brain panel during human play.
- Phase 8: deliver full training instrumentation, freeze/inspect controls, and integration with Watch Fly.
- Phase 10: verify every model module is reachable in the inspector, sampled tensor values match backend captures, selected moves match displayed policies/search data, and stale positions are never shown as current.

Telemetry must be optional, bounded, rate-limited, and independently switchable for activations, search, and gradients. Transfer only sampled detached summaries to the GUI; large weight views load on demand. Measure memory use and training/search throughput with telemetry on and off on the target machine, record the overhead, and tune defaults accordingly. Turning visualization off must not change the learning algorithm. MAX watch speed suppresses expensive visualization refreshes.

Visual acceptance: review real screenshots of menu, play with brain panel, training with brain panel, layer drill-down, search drill-down, settings, and empty/error states at the minimum and a larger supported window size. Check monochrome legibility, no overlaps/clipping, clear chess interactions, and a consistent authored appearance.

## Required experimental feature — temporary opponent adaptation

During a human match, Fly builds an ephemeral model of the opponent's observed move choices and may adapt its search strategy within that same match. This is an optional play feature, separate from persistent self-play learning. It must be presented as an estimate from limited evidence, never a diagnosis of the human or a guaranteed strength improvement.

- Phase 3 reserves a search-policy extension boundary for opponent priors. The baseline search and value perspective remain independently testable.
- Phase 7 implements a session-scoped opponent profile. Update it only from actual human decisions and the legal alternatives available in that position. Begin with interpretable features such as capture/check preference and willingness to exchange, conditioned on opportunities. Avoid inferring a preference simply because a move was forced.
- Use a regularized, small online model over legal move features, with the network policy as its prior/baseline. Start from the baseline, bound adaptation strength, and reduce influence with little evidence. Do not fine-tune the main network on a handful of human moves.
- Only information available when the human move was played may inform its prediction. Record prediction before observation for honest scoring. Never use future moves or the game's future result to explain earlier decisions.
- At opponent search nodes, an experimental bounded blend may allocate exploration toward predicted human replies. Retain baseline exploration of all legal replies; do not replace adversarial value backup with an assumption that the human will cooperate. Reject any design that suppresses critical defenses solely because they seem unlikely.
- Expose an Adapt To Me toggle and a compact brain-panel view: observed decision count, estimated tendencies, baseline versus adapted reply probabilities, and uncertainty/insufficient-evidence state. Clearly distinguish opponent prediction from Fly's own value and policy heads.
- Reset on new game or opponent change. Undo rebuilds the profile from the surviving human-move history. An asynchronous search uses an immutable profile snapshot tagged with the game and request IDs. Nothing is written to the main checkpoint or shared across opponents by default.
- Phase 8 integrates this profile into the live brain inspector. Self-play training and model-promotion evaluations keep it disabled unless an explicitly labeled experiment requires it.
- Phase 10 measures prediction accuracy and chess outcomes against fixed scripted opponent styles using matched seeds/positions with adaptation on/off. If no benefit is established, retain the experimental label and leave it disabled by default.

Acceptance: profile reset/undo correctness; forced-move handling; no update from Fly's own moves; no main-weight/checkpoint changes; normalized legal predictions; bounded influence; stale snapshot rejection; honest uncertainty; reproducible enabled/disabled comparisons. This feature cannot promise to understand a player's strategy from one short game.
