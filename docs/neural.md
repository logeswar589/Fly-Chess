# Neural contracts — Phase 2

## Network

`PolicyValueNetwork(NetworkSpec(channels=32, residual_blocks=2))` uses 70,437 parameters with the default specification. The stem is a 3x3 convolution, GroupNorm, and ReLU; each residual block has two 3x3 convolutions. GroupNorm has no running batch statistics, so one-position inference and replay batches share the same normalization rule.

Input is float32 `[batch, 21, 8, 8]`. Outputs are unnormalized policy logits `[batch, 20480]` and tanh value `[batch]`. The policy projection gives each source square 320 destination/promotion logits, then permutes rank/file ahead of destination/promotion to preserve the Phase 1 action mapping. Value is expected outcome from the player-to-move perspective, not centipawns or a calibrated win probability.

The default is deliberately small. Neither random initialization nor successful optimizer tests establish playing strength. Full losses, replay, and the self-play training loop remain future phases.

## Inference and ownership

`Evaluator(model, model_id=...)` switches an owned model to eval mode. It runs batches under `torch.inference_mode()`. Pass complete `ChessGame` objects so draw-claim policy and repetition history are retained. Terminal positions bypass the network and return exact results with zero policies. Empty batches return an empty list. Invalid legal logits or value outputs fail explicitly.

`masked_policy(logits, masks)` masks before stable softmax, sets illegal actions to zero, and returns zero terminal rows. Nonfinite illegal logits are irrelevant and masked out; nonfinite legal logits are an error. `PolicyAgent` provides deterministic argmax or seeded temperature sampling over legal actions for experiments, without search-improved targets.

An evaluator and its network belong to one worker. Do not concurrently train, mutate, or inspect that same model from another thread/process. Future training/search workers must use separate frozen snapshots with their own model IDs. Position IDs include the root FEN, move history, current position, and draw-claim policy. UI game/request IDs must additionally distinguish separate games with identical positions.

## Inspection

`module_inventory` lists every named module (including residual/container modules) and its own parameters, avoiding double-counted totals. `evaluate([game], inspection=InspectionRequest(request_id))` captures one position only. Default sampling allows 4,096 total scalar values and at most 64 per output tensor; both limits have a hard ceiling of 65,536. Each tensor reports its true full shape, flattened prefix values, and truncation status. It is a bounded prefix sample, not a full heatmap or a statistical distribution.

Select one named module and raise the sampling budget when a future inspector needs a full small activation tensor. All module output shapes remain listed even when their data budget is exhausted. Metadata includes model ID, position ID, FEN, side to move, request ID, UTC capture timestamp, and sampled status. Input planes are separately available through the versioned encoder and PLANE_NAMES.

Hooks are scoped to the forward call and removed even on error. Data is detached and copied into ordinary CPU/Python values; no autograd graph is retained. Inspection does not alter the network output or RNG stream. There are no persistent hooks or telemetry queues. A future GUI must rate-limit requests, reject stale request IDs, and measure telemetry overhead. Gradient/weight-change diagnostics arrive with the training loop.

## Reproducibility and devices

Call `seed_everything` once in each owning process before model creation/device selection. It seeds Python, NumPy, and PyTorch, limits CPU threads, disables cuDNN benchmarking, and optionally requires deterministic operations. A process-local `PolicyAgent` owns its separate NumPy generator. Full RNG checkpointing arrives in Phase 5.

Repeatability is tested within the same software/hardware configuration; identical results across GPU models, CPU/CUDA, or library versions are not promised. Unsupported deterministic operations fail instead of silently becoming nondeterministic. Reference: [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Weight files

`save_weights` writes a same-directory temporary file, flushes it, then replaces the target atomically. Files include network specification, model identity, state dict, and encoding/action/network/format versions. `load_weights` uses restricted `weights_only=True`, CPU staging, schema validation, strict parameter loading, and finite-value checks. A failed write preserves the prior target. Loading constructs a separate model without consuming the CPU RNG stream.

These are inference files, not resumable checkpoints. Only load files from a trusted source; restricted loading is not resource-exhaustion protection. The full checkpoint manager will add optimizer, scheduler, progress, replay references, and RNG states in Phase 5.
