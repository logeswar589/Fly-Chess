# Fly in your browser

The browser edition runs the same Python chess rules, neural inference, MCTS,
opponent adaptation, archival and training workers as the desktop app. Chrome,
Edge and other modern browsers display the interface. It uses local assets and
requires no JavaScript build step, CDN, account or extra Python dependency.

## Start on this computer

Double-click `Start-Fly-Web.cmd` in the project folder, or run:

```powershell
cd "D:\Ai\projects\FlyBrain chess"
.\.venv\Scripts\python.exe -m fly_chess --config configs/lightweight.toml web
```

Open the **full access link printed in the terminal** in Chrome. The default
address is `http://127.0.0.1:8765/`; its `#token=...` fragment grants access to this
server instance. The browser stores the token for its current tab session and
removes the fragment from the address bar. A fresh server creates a new token.
The page also accepts the token through its connection form.

Keep the terminal running. Closing the browser leaves training running. Ctrl+C
in the terminal closes the server, cancels play search and waits for training to
save at a safe boundary. A long self-play game can delay that boundary.

## Another device

```powershell
.\Start-Fly-Web.cmd --host 0.0.0.0
```

Open the printed **LAN** access link on a phone, tablet or another computer on
the same trusted network. Windows may ask to allow Python through the firewall;
choose the private network if you want LAN access. The host PC must remain on.
All connected browsers control the **same match and training workspace**.

For access outside your home network, use a private VPN to the host and its VPN
address with the same port/token. Localhost by itself is not internet hosting.
This built-in HTTP server is intended for personal local/trusted-network use,
not direct public port forwarding or multi-user public deployment. Public hosting
would need HTTPS, persistent authentication and deployment hardening.

## Play and inspect

Choose side, search budget and a workspace checkpoint, then Begin a new match.
Click/tap a piece and destination. Promotion offers all four choices. Undo, board
flip, draw claims and resignation use the same rule engine as desktop play.
The match retains its chosen weights until a new match starts.

The Brain canvas is interactive:

- Drag to rotate; scroll/pinch or +/− to zoom between 0.6× and 4×.
- The arrow in the corner expands the brain to fill the page; Escape closes it.
- Hover over a neuron sample to reveal its layer, flat tensor index and activation.
- Synapses toggles sampled weighted connections. Connected edges brighten on hover.
- Freeze pins the measurement. Reset restores the camera.
- Keyboard focus on the canvas supports arrow rotation and +/− zoom.

The anatomy-inspired coordinates are an illustrative arrangement of Fly's
artificial network, **not a reconstructed fly connectome**. The nodes are measured
activation samples. Edges are real Conv2d/Linear terms, sampled at up to twelve
output units per layer: each uses the strongest absolute weight in that output's
kernel/row. Convolution edges use the actual input receptive-field coordinate;
padding-only terms are omitted. Edge brightness follows `abs(input * weight)`.
These are selected contributions, not the complete connectivity or a causal
explanation; biases, other weighted terms, normalization and residual additions
are not represented by these edges. Node magnitude is normalized per layer and
hollow nodes represent negative activations. Values do not pulse artificially.

## Training and replay

Train offers 1/2/5/10 additional generations, Start/Resume, Load, Pause, Save and
Stop. Existing checkpoints resume their saved configuration and verified replay.
Keep the workspace's models and data directories together. Training diagnostics
default to every optimizer update in the browser; Settings offers 1/8/32 intervals
and independent activations/gradient controls.

Before/After uses the same sampled replay position around an actual optimizer
update with shared per-layer brightness scales. The latest sample remains on
screen during self-play/evaluation. There is **no live self-play search stream**;
the empty-state message identifies when a new optimizer capture is needed.
Loading a checkpoint restores training state but does not fabricate old captures.

Watch replays real saved games at ¼×/½×/1×/2×/MAX; its brain is recomputed with the
exact archived model. MAX suppresses activation refresh. Stats show actual recent
losses and evaluation status; absolute strength remains unrated. Full desktop
module heatmap/search-table controls are still available in the Python GUI.

Settings controls temporary opponent adaptation, play telemetry, opt-in local
human-game saving and explicit human-data training. Human training produces a
separate candidate through the existing evaluation gate. It does not stream the
self-play optimizer's Before/After captures.

## Access boundaries and verification

API reads and writes require the access token; cross-origin mutations are rejected.
Only listed static assets are served, and model selection is restricted to the
workspace's listed checkpoints. Replay files, weights, logs and arbitrary paths
are not exposed as downloads. Access links grant control of the shared workspace;
share them only with intended users. This is a single-user lab interface.

Automated HTTP tests exercise unauthorized requests, cross-origin writes, path
restrictions, legal moves/AI replies, stale move rejection, underpromotion, real
training, checkpoint reload and replay. Browser checks cover the actual interface,
brain zoom/rotation/expansion and responsive layout.

Visual direction: the dark editorial palette, serif headlines and spacious
composition draw inspiration from [Elyse Residence by Phenomenon Studio](https://dribbble.com/shots/25815603-Elyse-Residence-Luxury-Real-Estate-Website-Design).
The chess graphics and interface implementation are original; no artwork was copied.
