# Opt-in human games and supervised candidates

Saving is **off by default on every launch**. Enable Save human games in Settings,
or Train → Human. When you choose New or close the application, a match with moves
is written atomically to `data/human_games/<game-id>.pgn` by a bounded background
archive worker. No PGN is written merely by playing while saving is disabled.
The toggle applies to the current surviving match history when leaving it.

The PGN includes legal moves/root position, colors, result, termination, explicit
consent, checkpoint path/model identity and whether temporary adaptation was used.
No player account or real name is collected. Positions/actions are reconstructed
from the PGN's legal main line. Undo removes abandoned moves before export.
An unfinished or abandoned game gets `*` and no outcome target. Resignation records
the actual conceded result; claimable draws are verified from the position/history.
Automatic results are checked against python-chess. Save errors stay visible and
retain the pending board snapshot for Retry save; normal exit waits for saves.

## Explicit learning

Train → Human → Train from human games is a separate action. Its base is the model
selected in Play setup, falling back to best/latest if available. It uses one epoch,
learning rate 0.0001, and at most 4,096 positions. The CLI supports bounded overrides:

```powershell
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml train-human --base models/fly_best.pt --epochs 1 --max-positions 4096 --learning-rate 0.0001
```

The objective is legal-move cross entropy on the **human's actual choices**, plus
player-to-move outcome MSE and the configured L2 penalty. Fly's own moves are not
imitation targets. This can learn a human's mistakes; it does not imply better
chess. Only completed, consented, validated app PGNs qualify. Incomplete games are
excluded, and missing consent or inconsistent outcomes fail explicitly. Generic
PGN import without the provenance fields is not currently supported.

Data is scanned in stable filename order up to the position cap. A capped final
game can contribute a subset of its human decisions, all with its verified final
outcome. Manifests report used/available positions and incomplete files encountered.
The immutable run manifest under `data/human_training` embeds the selected PGNs,
their hashes, source model identities and sample game/ply keys. It is separate from
the self-play SQLite database. Human training never inserts into self-play replay
or changes `fly_latest.pt`.

A new inference-only candidate goes under `models/human_candidates`, with parent
hash, dataset checksum, objective and update history. It cannot resume the self-play
optimizer; continue self-play from its existing full latest checkpoint. Human
training has Stop, not optimizer-resume: stopping during updates saves a clearly
partial candidate with no evaluation/promotion. Restart creates a new run.

The candidate always uses the established evaluation gate, even when automatic
self-play generation evaluation is disabled. If no best exists, the unchanged
parent becomes the initial baseline, then the human candidate competes against it.
Promotion requires all requested paired matches to complete, sufficient pairs, and
the existing conservative score bound above 50%. A human training loss reduction
does not bypass that gate. Interrupting evaluation preserves the candidate and its
journal; the regular `evaluate --candidate ... --id <human-run-id>` command can
resume with the same evaluation settings. Play candidate explicitly tries a
completed candidate even if it was not promoted.

If a human candidate is promoted, best remains an inference-only file; self-play
resume still uses the separately maintained full latest/generation checkpoint.
Training manifests, PGNs and candidate files are not automatically pruned.

## Verification

Phase 9 full suite: 136 passed. Tests cover both human colors, PGN round-trip,
consent, incomplete/undone/resigned games, malformed provenance/outcomes, archive
failure/retry, separate datasets, changed candidate weights, unchanged incumbent,
partial cancellation, and real gate rejection of insufficient truncated matches.

`scripts/verify_human_ui.py` runs an isolated fixture game, opt-in archive, actual
button dispatch, one supervised update and real evaluation while rendering frames.
The verified run under `logs/human-ui-8d56538e` used two human positions, rendered
266 frames, and retained the incumbent. This proves the workflow, not strength.
