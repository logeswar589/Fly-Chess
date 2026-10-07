# Training and recovery — Phase 5

## Generation pipeline

`TrainingController(workspace, config)` creates a new run; an existing training run must be resumed explicitly. `run(generations=N)` completes N generations relative to the number already completed, including the unfinished current generation first. Self-play uses a frozen snapshot of the current model. Once the configured games finish, optimization samples replay minibatches and performs the configured updates. The Phase 6 evaluation gate then compares the frozen candidate with best before the controller advances the generation.

The replay used by training is `data/replay_buffer/training.sqlite3`, separate from standalone Phase 4 diagnostic replay. Fresh training does not silently ingest earlier experimental datasets. Both rule-completed and explicitly marked ply-truncated games are used; truncation targets are zero by the documented convention. Aborted games are not training data. An undersized nonempty replay produces a smaller batch; empty replay fails clearly.

The objective is mean legal-policy cross entropy against stored search targets, plus mean squared player-relative value error, plus `weight_decay * sum(parameter**2)`. Here `weight_decay` names the explicit L2 coefficient; Adam has no additional weight decay. Policies are normalized over each sample's legal actions for the loss. Gradients are checked and clipped before Adam steps; nonfinite loss/gradients/weights stop training. ExponentialLR advances after each optimizer step. Each persisted metric identifies the actual batch size, learning rate used, losses, gradient norm, generation/step, and replay sample IDs.

`training_inspection_interval=0` disables learning inspection. A positive interval captures actual per-module gradient/weight/update norms and the first sampled position's target versus predicted legal policy/value before and after that same update. Values are detached and serializable. A single update is not claimed to improve chess strength.

## Controls and ownership

The controller owns a workspace lock for its lifetime. Use it as a context manager, and run it on the future training worker rather than the GUI thread. Other threads can call `request_pause()`, `request_stop()`, and `request_save()`; they only set events. The owning worker acknowledges pause/stop by returning after saving at a consistent boundary. Calling `run()` again continues a paused/stopped controller. Close only after the worker returns.

Controls finish the current self-play worker wave or minibatch, so they are safe but not instantaneous. A wave has at most the configured worker count of games. Manual save is acknowledged when its event clears after a successful save. Public `save()` is for the owning worker only. After an exception, reload the last durable checkpoint instead of continuing the failed in-memory controller. Ctrl+C exits with a resume message, preserves the previous checkpoint, and joins spawned workers; an unfinished game may restart.

## Checkpoint contents

`models/fly_latest.pt` is atomically replaced at each completed game, minibatch, stage transition, and control acknowledgment. `models/fly_generation_NNN.pt` is additionally written at the configured generation interval. Completed-generation files point to the next generation's start; a mid-generation latest file records that generation and its exact next operation. `fly_best.pt` is created or replaced only by the Phase 6 gate, with the first model explicitly labeled as a baseline.

Full checkpoints contain model weights/specification, optimizer moments, scheduler, configuration, progress/stage/counters, actual metrics, RNG states (Python, NumPy global and replay sampler, torch CPU and available CUDA devices), and a replay manifest. Absolute Elo stays None; Phase 6 evaluation summaries carry only supported relative estimates. Self-play White/Black outcomes are labeled accordingly. `active_seconds` measures active controller work rather than paused wall time; it is not a compute benchmark.

The full files also support the existing inference weight loader. Conversely, inference-only files lack training state and cannot be used for resume. Checkpoints are loaded with `weights_only=True`, schema/required-field/finite-state validation, strict model state loading, and replay checksums. Only load trusted files; restricted unpickling does not bound memory use of a malicious file.

## Durable replay and recovery

Every replay change creates an immutable SQLite backup under `data/replay_buffer/snapshots`. Its relative path and SHA-256 hash are included in the checkpoint. Optimization-only checkpoints reuse the unchanged replay snapshot. Resume validates the snapshot before replacing the active replay; it therefore cannot silently pair older weights with a newer or evicted sample population. Missing/corrupt snapshots fail with an explanatory error.

Games have stable run-ID/index identities. An archive is saved before replay ingestion and the checkpoint's completed-game marker. If a crash occurs in that gap, resume restores the prior replay snapshot, validates the expected archive/model/seed, and imports that game idempotently. The next index advances only with its checkpoint. A completed minibatch's weights, optimizer, sampler RNG, metrics, and progress commit together. If replacement fails after the optimizer applied its in-memory update, restart loads the prior boundary and redoes that update exactly once.

Saving uses same-directory temporary files, flush, and atomic replace. A failed replacement leaves the prior checkpoint intact. Replay snapshots and archives remain available for older checkpoints. There is currently no automatic pruning of snapshots, archives, or accumulated metrics; disk usage grows. Preserve the workspace data and models together when moving a run. A single `.pt` file alone is sufficient for inference, but not for full training resume.

## Reproducibility and limits

Saved configuration is authoritative on resume; conflicting overrides are rejected. The resolved device can depend on availability when `device=auto`. Exact same-device/software continuation is tested; cross-device or different-library numerical equality is not guaranteed. CUDA RNG restoration requires matching device counts when CUDA is present. Individual game seed streams advance across generations. Worker-side RNG resets are isolated from the parent's replay/training RNG at commit boundaries.

Recovery tests cover a partially finished generation, interrupted checkpoint replacement after a game and after an optimizer step, exact next-update comparison against uninterrupted CPU training, fresh-process resume, CUDA training resume, duplicate prevention, schema/corruption rejection, and workspace ownership. Small smoke runs verify the pipeline and real parameter updates; they do not establish improving Elo or practical strength.

