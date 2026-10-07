# Desktop interface and visual specification

Launch `fly-chess gui`, or `fly-chess gui --weights path/to/model.pt`. The global
`--workspace` option precedes `gui`. A training checkpoint also works for inference.
Without a checkpoint, the menu explicitly offers an untrained network. No chess
strength is implied. Drop a `.pt` file to start a new match with that model.

## Visual system

The 1440×960 reference canvas scales and letterboxes in a resizable window with a
1200×800 minimum. Text and drawing primitives render directly at the window's
resolution; Windows DPI awareness prevents operating-system bitmap enlargement.
Near-black #121212, warm white #efeee9, neutral gray labels and
thin rules form the palette. The board uses light #d9d8d1 and gray #666763 squares.
Original vector silhouettes have contrasting outlines on either square color.
Bundled Manrope and IBM Plex Mono fonts include their OFL licenses. All assets
load relative to the installed package, not the current working directory.

The 608px board stays primary. Brain inspection occupies the right column.
Selection uses a dark outline, legal destinations a dot/ring, last move a light
inset border, and check a double dark border. Movement takes 160ms and can be
disabled in Settings. No decorative animation or fabricated training metrics.
`scripts/render_ui.py` produces reviewable screenshots under `logs/ui-preview`.

## Playing and inspecting

Click a piece and its destination, or use arrow keys and Enter. Promotion offers
all four legal choices. Tab/Shift+Tab and Enter navigate controls. Escape clears
selection. Undo restores the previous human decision and invalidates any pending
AI move; when playing Black it preserves Fly's initial move. Claimable draws are
explicit, automatic chess outcomes remain automatic. No chess clock is exposed.

| Budget | Simulations | Soft seconds | Sampling temperature |
|---|---:|---:|---:|
| Easy | 16 | 1 | 0.5 |
| Medium | 64 | 2 | 0.15 |
| Hard | 128 | 3 | 0 |
| Full power | 256 | 5 | 0 |

All budgets cap nodes at 4,096 and depth at 128. An in-flight evaluation may finish
beyond the time limit. Budgets describe work, not established playing strength.

Overview shows the implemented inference/search path and an uncalibrated value
from the source position's player-to-move perspective. Layers reaches every
network module, its own parameter count, selected activation channels, input
planes, and bounded parameter slices. Tensor cycles multiple outputs/parameters.
Each heatmap uses its own symmetric scale; flat slices are labeled explicitly.
Captures include at most 65,536 activation values and 4,096 values per selected
parameter. They do not claim to display every scalar weight simultaneously.

Search shows raw legal priors, visit shares, parent-perspective Q, selected move,
principal variation and at most 512 tree edges. Candidates and tree are paginated.
Clicking a candidate draws an overlay only when its source FEN matches the board.
The board remains the current game even when inspecting a historical sample.
Freeze pins the sample; < and > inspect the last 16 recorded Fly decisions.
Source ply, side, model identity and position hash remain visible. Overview gives
the capture timestamp. Frozen layer changes are disabled because those activations
were not recorded for another layer. Telemetry is optional in Settings.

Inference runs on one owning worker, with one outstanding job and a four-message
bounded queue. Progress samples occur every eight simulations. Game/revision
tokens reject stale results after undo, new game, settings or model changes.
Errors are visible in the footer. Window close cancels search and joins its worker.

## Experimental opponent adaptation

Off by default. A regularized three-feature move-choice model conditions on legal
capture, check and central-destination opportunities. It updates only from human
moves, scores predictions before observing them, and ignores forced/no-variation
moves for learning. Influence is `0.2*n/(n+12)` for informative decisions, always
below 20%. It blends only opponent-node exploration priors; adversarial backup and
baseline exploration remain intact. No persistent chess weight or checkpoint is
modified. Undo reconstructs surviving history; new match resets it. The display
labels sparse evidence and shows baseline/adapted replies on the human turn.
There is no measured strength benefit yet.
