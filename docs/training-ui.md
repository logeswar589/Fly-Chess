# Graphical training and observation

Run `fly-chess --config configs/lightweight.toml gui` for the desktop app. For a
short verification run use a **new** workspace:

```powershell
.\.venv\Scripts\python.exe -m fly_chess --workspace logs/my-gui-smoke --config configs/smoke.toml gui
```

Train opens the training lab. Start/Resume runs the selected number of additional
generations (1, 2, 5 or 10). Pause, Stop and Save set cooperative events. They finish
the current self-play wave/minibatch; evaluation can stop between searches. Close
keeps the window responsive while waiting for a safe checkpoint. No forced worker
termination is used. Long default games/evaluations can make this wait substantial.

Load reads `models/fly_latest.pt`, or drop a full checkpoint onto the training
screen. Loading is read-only; Resume restores the saved configuration and replay
snapshot. Keep the checkpoint's workspace data with it. An inference-only `.pt`
cannot resume self-play training. Drop a TOML file before starting a new run to
choose budgets. Active runs reject checkpoint/config changes. Telemetry options
can change at boundaries and do not change the saved training configuration.

Training runs in a spawned process, isolated from Pygame and play inference RNG.
Three detached reports and two pending settings messages bound IPC queues. Slow
viewers can miss intermediate display updates; game archives, replay and checkpoints
remain durable. Charts use the latest 1,000 persisted update/evaluation records,
with an explicit omitted count. No callback draws or handles events in a worker.

## Brain and charts

The Brain tab shows a zoomable, rotatable fly-inspired view of measured activation
samples across all leaf modules. Before/After compares the same replay position
around a real optimizer update, using shared per-layer brightness scales. Hover
shows the exact source tensor index and value. Freeze pins the update. Sampling
uses the existing diagnostics/activation switches and 1/8/32-step interval. During
self-play or evaluation, this view retains the latest sampled optimizer update;
it does not claim to stream those workers' live search activity. Old checkpoints
without brain captures show an empty state until a new sampled update arrives.

Learning samples one real replay position at a configurable 1/8/32-step interval
(default 8). Freeze retains it as later updates arrive. The board, sample game/ply,
optimizer step, side and capture time identify the source. Before/after views use
the same position and shared symmetric heatmap scale. A change is not a promise
of better play. Module controls choose the next sample; they do not invent missing
captures from prior updates. Tensor cycles captured outputs and selected weights.
Weight slices have a histogram, and the learning view shows clipped gradient,
pre-update weight and update norms where a module owns parameters. Activations and
gradient inspection have independent switches, plus a master diagnostics switch.

Captures are bounded to 65,536 activation values per forward and 4,096 values per
selected parameter. Large arrays remain transient, not accumulated in checkpoints.
Policy/value targets, predictions, norms and provenance remain in persisted update
metrics. A missing optional game archive disables the source-board/watch display
without invalidating a valid replay optimizer step.

Stats displays actual total/policy/value losses, candidate win rate, cumulative
mean game length, true outcomes, separately counted truncations, learning rate and
saved active time. Missing evaluations remain gaps. Absolute rating remains
Unrated, because changing-incumbent relative comparisons do not establish a human
Elo curve. Stage labels on idle runs identify the **next** operation.

## Watch Fly

Watch replays actual archived self-play games, labeled recorded/completed/truncated.
Speeds are 0.25×, 0.5×, 1×, 2× and MAX; 1× advances one ply per second. Pause and
single-step work in both directions. Only the current and latest pending game are
retained. Playback never changes training results or blocks training progress.
MAX jumps to the last position and suppresses expensive inference refreshes.

At other speeds, a separate inference worker recomputes bounded activations using
the exact archived self-play model. It rejects stale game/position updates. The
panel distinguishes current and previous recorded-position samples. Original
self-play search trees were not archived and are not fabricated during playback.

## Verification and measured limits

Phase 8: 128 tests passed on Windows, including process pause/save/exit/reload/resume,
exact weights with diagnostics on/off, independent gradient toggle, watch speeds,
missing chart values and the existing CPU/CUDA recovery tests. The real offscreen
GUI smoke workspace `logs/gui-training-a2f4b2d2` completed two eight-ply games and
two updates while rendering 309 frames: mean frame work 3.93ms, maximum 29.49ms.
This checks event-loop work, not monitor refresh latency or a long strength run.

RTX 3050 / CUDA microbenchmark, five warm trials, default 32-channel/2-block model:
32-simulation search about 153–159ms with/without sampled telemetry (difference
within run-to-run noise); a batch-of-eight update 24.45ms without vs 41.61ms with
full diagnostics every step. Hence the default samples every eight updates. Extra
traced Python allocation peaked around 45KB/200KB for updates without/with capture;
CUDA total allocated peak was about 73MB in that benchmark process. These exclude
disk/checkpoint/render costs and are not overall training throughput estimates.
Reproduce using `scripts/benchmark_telemetry.py` after a GUI smoke run.

`scripts/verify_training_ui.py` creates a unique smoke workspace, runs actual
training with an active render/input loop, and saves 1440×960 and 1200×800 screenshots
of learning, stats, config and watch. Main project models are not trained by it.
