# Fly-Chess 0.1.0 — local release

The ten implementation phases are complete. This release provides legal neural
chess, self-play training and recovery, guarded model competition, the monochrome
desktop interface, actual network/search/learning inspection, recorded Watch Fly,
experimental temporary opponent adaptation and opt-in human-data training.
It does not bundle a strength-trained model or claim a measured Elo.

## Install and launch

The source archive includes the application, fonts/licenses, configuration presets,
tests, scripts and documentation. It excludes environments, logs, game datasets and
model checkpoints. Extract it, open PowerShell in the extracted directory, then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fly_chess gui
```

Alternatively install the wheel into an existing Python environment:

```powershell
python -m pip install .\fly_chess-0.1.0-py3-none-any.whl
python -m fly_chess --workspace .\fly-workspace gui
```

The wheel includes fonts and licenses. Python dependencies are resolved by pip and
are not bundled into the wheel. This is a Python desktop release, not a standalone
Windows executable. `Start-Fly.ps1` launches the source checkout using its `.venv`
and defaults storage to the script directory, including when called elsewhere.
If PowerShell script execution is restricted, use the Python launch command above.

`constraints-tested.txt` records the exact tested Windows/Python 3.14.5 versions.
It can be supplied as `-c constraints-tested.txt` during setup on that Python version.
CPU inference/training and available CUDA execution are tested. Use the CUDA setup
command in README for the tested RTX 3050 configuration. Other operating systems,
Python versions, drivers and display scaling configurations were not release-tested.

## Verified workflows

- Full regression: **139 passed in 52.86 seconds**, including CPU and CUDA recovery.
- Source workflow: train one generation → play White → close all workers → load in
  a new app → resume a second generation → run candidate competition → play Black
  with the newly saved model. Four self-play games, four optimizer steps, four
  evaluation games. Capped games remained truncations and did not trigger promotion.
- Fresh wheel target: the same complete workflow ran outside the repository's
  working directory, asserting every app import came from the newly installed
  wheel target. Existing verified Python/native dependencies were reused; this was
  not a clean-OS or offline dependency installation test.
- Both workflows reported zero surviving registered child processes after close.
- Every network module was requested through the real GUI worker. Input planes and
  parameter samples matched the backend exactly; activation captures stayed bounded
  and left no hooks attached. Same-match weights remain frozen; a new match reloads
  a replaced checkpoint instead of silently retaining the old cached network.
- Real Pygame renders cover menu, play, layers, search, training, watch, settings,
  empty and error states, including 1200×800, 1440×960 and 1920×1280 outputs. A
  timestamp/title overlap found during review was corrected. Rendering checks used
  SDL's offscreen driver, not OS-level native-window interaction automation.

Verification tools:

```powershell
.\.venv\Scripts\python.exe scripts/verify_release.py
.\.venv\Scripts\python.exe scripts/build_release.py
.\.venv\Scripts\python.exe scripts/check_package.py
```

Build produces a wheel, a source ZIP and `dist/manifest.json` with SHA-256 checksums.
The package check creates a unique installation/workspace under logs. Existing
user training state is not overwritten by these scripts.

## Performance and experimental adaptation

Test machine: Windows, Python 3.14.5, PyTorch 2.11.0+cu128, NVIDIA RTX 3050 Laptop
6GB, driver 616.92, Pygame CE 2.5.8. The source integrated run rendered 2,312 frames
while training/search ran separately: mean frame work 3.46ms, maximum 25.54ms.
The first wheel run measured 3.33ms mean and 36.78ms maximum over 2,262 frames.
These measure event-loop work, not monitor latency or guaranteed frame rates.

The earlier default-network CUDA microbenchmark measured about 153–159ms per
32-simulation search and 24.45ms/41.61ms per batch-of-eight update without/with full
diagnostics every step. Sampling every eight steps remains the GUI default.
Warm kernel timings exclude disk/checkpoint costs; short smoke throughput must
not be extrapolated into an Elo or long training-time promise. Full default
evaluation alone requests 100 games, and long self-play games can take substantial
time. Keep budgets finite and inspect actual progress before scaling them.

`scripts/verify_adaptation.py` ran four matched-seed on/off pairs against two fixed
scripted move styles, with both human colors, eight simulations and a 32-ply cap.
All eight games were truncated; main weights were unchanged. Pre-move baseline and
adapted prediction losses are recorded in `logs/adaptation-experiment.json`.
This small untrained-network diagnostic does not establish better chess outcomes.
Adaptation remains optional, experimental and off by default.

## Recovery and known limits

- Use only trusted checkpoints. `weights_only=True`, strict schemas and checksums
  reduce parsing errors but do not make arbitrary files safe or bound their memory.
- Keep models and replay snapshots/data together for full resume. Inference-only
  weights and human candidates cannot resume the self-play optimizer. If a human
  candidate becomes best, full latest/generation checkpoints remain separate.
- Pause/stop/close wait for a safe game-wave/minibatch boundary; an in-flight neural
  operation may finish beyond a soft search deadline. Do not force-kill the process
  if you need the current operation committed. Earlier durable boundaries remain
  recoverable after interruption.
- Human saves are opt-in. Failed saves keep their pending snapshots for Retry save;
  closing rejects late AI results so the archived line cannot lag a new AI move.
- Replay is bounded, but game archives, immutable replay snapshots, checkpoints,
  human manifests and persisted metric history are not automatically pruned.
  Disk usage grows; keep backups and do not delete snapshots referenced by retained
  checkpoints. Long-run multi-hour resource behavior has not been characterized.
- Layer/weight views are bounded slices, not simultaneous drawings of every scalar.
  Watch is recorded playback with recomputed activations, not a fabricated live
  self-play search tree. Missing ratings/evaluations remain missing.
- One python-chess dependency deprecation warning occurs on Python 3.14 concerning
  the future removal of an asyncio policy API. It does not fail the tested workflows.

The release establishes functionality and recovery in the tested scenarios.
Meaningful playing strength requires a separate sustained training and evaluation
campaign; the included verification checkpoints are deliberately small smoke models.
# Browser follow-up

The source archive now also includes Start-Fly-Web.cmd and docs/browser.md. The
wheel bundles the web HTML/CSS/JavaScript. `fly-chess web` starts a token-protected
local browser interface; add `--host 0.0.0.0` for a trusted LAN. Current regression:
150 tests passed. Browser assets and weighted brain connections are additional to
the original desktop release verification below.
