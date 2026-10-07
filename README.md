# Fly-Chess

A phased project for a trainable self-play chess agent, a bespoke monochrome interface, and live inspection of its network and search.

**Local release 0.1.0: all ten implementation phases complete.** Play in the monochrome desktop interface, inspect actual network/search data, and train with pause/save/resume, recorded self-play playback, learning captures and persisted charts. Optional temporary opponent adaptation and human-game saving are off by default. Verification establishes functionality, not chess strength. See [release verification and limitations](docs/release.md).

## Windows setup

Python 3.11+ is required. Tested on Windows with Python 3.14.5 and Pygame CE 2.5.8.

Release artifacts are `dist/fly_chess-0.1.0-py3-none-any.whl` and
`dist/fly-chess-0.1.0-source.zip`, with checksums in `dist/manifest.json`.
The source launcher is `Start-Fly.ps1`; the Python command below works directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml diagnose
.\.venv\Scripts\python.exe -m fly_chess gui
.\.venv\Scripts\python.exe -m pytest
```

For the detected RTX 3050 laptop GPU, the CUDA 12.8 PyTorch build can be installed in the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
```

On other machines, choose the matching build through the [official PyTorch installer](https://pytorch.org/get-started/locally/). `device = "auto"` probes actual CUDA execution and falls back to CPU with a reason; explicit `cuda` fails clearly if unusable. No driver or system toolkit installation is performed by Fly-Chess.

## Desktop play, training, and human learning

For graphical play, use `fly-chess gui --weights models/fly_best.pt`, or omit weights
for the honest untrained/empty state. Train opens real training controls; Learning,
Stats, Watch, Config and Human select the lab views. See [UI controls](docs/ui.md)
and [training UI](docs/training-ui.md) for checkpoints, inspection, playback and limits.

Play and Train also have a **Brain** tab: scroll to zoom, drag to rotate, and hover
to inspect real activation samples in a fly-inspired 3D layout. Training offers
Before/After views of sampled optimizer updates. The layout is illustrative;
the activity values come from Fly's actual artificial neural network.

Human-game saving is explicitly opt-in in Settings or Train → Human. Records save
on New/Exit; incomplete games never become outcome targets. **Train from human
games** creates a separate supervised candidate and evaluates it before any
promotion. See [human learning](docs/human-learning.md).

```powershell
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml train-human --base models/fly_best.pt
```

## Neural diagnostic

```powershell
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml neural-diagnose
.\.venv\Scripts\python.exe -m fly_chess neural-diagnose --inspect
.\.venv\Scripts\python.exe -m fly_chess neural-diagnose --save-weights models/diagnostic.pt
.\.venv\Scripts\python.exe -m fly_chess neural-diagnose --weights models/diagnostic.pt
```

These commands show legal move probabilities and player-to-move value. `--inspect` includes real module names, shapes, and bounded samples, not a rendered brain view. Saved diagnostic weights contain no optimizer, replay, or generation state; use the train command for resumable training checkpoints. Saving random weights does not make a trained model. See [docs/neural.md](docs/neural.md) for interfaces and inspection limits.

## Search diagnostic

```powershell
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml search-diagnose
.\.venv\Scripts\python.exe -m fly_chess search-diagnose --inspect --seconds 2
.\.venv\Scripts\python.exe -m fly_chess search-diagnose --self-play
```

Search reports a legal selected move, actual visits, root candidates, and stop reason. `--self-play` adds exploration noise; it does not start a training loop. `--inspect` includes up to 128 search edges. `--weights PATH` loads saved inference weights. Set `mcts_enabled = false` in a TOML config for explicitly labeled policy-only experiments. Time limits are cooperative and may overrun one inference call. See [docs/search.md](docs/search.md) for budgets, value perspectives, and the opponent-profile extension boundary.

## Generate self-play data

```powershell
# Short two-worker pipeline check: two games, capped at eight plies each.
.\.venv\Scripts\python.exe -m fly_chess --config configs/smoke.toml selfplay
# Longer finite batch, using the configurable lightweight budgets.
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml selfplay
```

Add `--weights models/your-weights.pt` to use saved weights or `--fen "FEN"` for a specific starting position. These commands write compressed games under `data/selfplay` and a SQLite replay store under `data/replay_buffer`; they do not train the network yet. The smoke preset counts ply-limit truncations separately from real draws. Repeated commands reuse the configured seed sequence with new game IDs; the training controller manages generation-wide seed/index progression. See [docs/selfplay.md](docs/selfplay.md) for data, worker, and recovery contracts.

## Train, save, and resume

```powershell
# Start one small verification generation in an isolated workspace.
.\.venv\Scripts\python.exe -m fly_chess --workspace logs/training-smoke --config configs/smoke.toml train
# Resume using the configuration saved in the checkpoint.
.\.venv\Scripts\python.exe -m fly_chess --workspace logs/training-smoke train --resume logs/training-smoke/models/fly_latest.pt
# Start a normal lightweight run in the current project (finite: one generation).
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml train --generations 1
# Continue that run later.
.\.venv\Scripts\python.exe -m fly_chess train --resume models/fly_latest.pt --generations 1
```

The smoke configuration uses short ply-limited games and two optimizer updates; it is a pipeline test. Training checkpoints include optimizer/scheduler, RNG state, progress, and a verified replay snapshot. Keep `models` and `data` together for resume. Existing runs are protected against accidental restart/overwrite, and only one training controller may own a workspace. Ctrl+C leaves the last durable boundary available for resume. The GUI exposes pause/stop/save at the same durable boundaries. See [docs/training.md](docs/training.md) for recovery guarantees, controls, and storage limits.

Each generation now runs the evaluation gate before completion. The first best model is labeled as an initial baseline; later candidates replace it only after a complete, statistically eligible match. Lightweight defaults use 50 color-swapped pairs (100 games), so evaluation can take substantial time. Smoke matches use smaller budgets and are explicitly insufficient for promotion. Reports live under `logs/evaluation`. Absolute Elo remains Unrated; only supported relative estimates are shown.

```powershell
# Evaluate a saved candidate separately; reuse the same ID to resume an interrupted match.
.\.venv\Scripts\python.exe -m fly_chess --config configs/smoke.toml evaluate --candidate models/fly_latest.pt --id trial-001
```

See [docs/evaluation.md](docs/evaluation.md) for promotion, confidence intervals, checkpoint behavior, and rating limits.

Activation is unnecessary. The diagnostic emits JSON and creates `data/selfplay`, `data/human_games`, `data/replay_buffer`, `models`, and `logs` under the current directory. Add `--workspace PATH` before `diagnose` to choose another storage root. Add `--fen "FEN"` after `diagnose` to inspect a position. Invalid configuration/positions exit with code 2 and an explanatory message.

If Windows denies access to pytest's shared temporary directory, use a new project-local test directory:

```powershell
$testScratch = Join-Path (Get-Location).Path ('.pytest_cache\verification-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $testScratch) { throw 'Expected a new test directory' }
.\.venv\Scripts\python.exe -m pytest --basetemp $testScratch
```

The core uses the `chess` distribution of python-chess for rule enforcement and NumPy for the network-ready representation. It does not invoke an external chess engine. No model weights or performance estimates are bundled.

See [PHASE_PLAN.md](PHASE_PLAN.md) for the full roadmap, [STATUS.md](STATUS.md) for the next step, and [docs/contracts.md](docs/contracts.md) for representation and draw semantics.

