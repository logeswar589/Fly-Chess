# Core contracts — version 1

Only standard chess is supported. `ChessGame` validates imported positions and rejects illegal moves. Use its history-preserving `copy()` for search; FEN alone loses repetition history. The underlying board is available for queries, but mutations should pass through the adapter.

## Action representation

20,480 slots, `(from_square * 64 + to_square) * 5 + promotion_code`.
Promotion codes: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen.
Squares follow python-chess: a1=0, h1=7, a8=56, h8=63.
Same-square slots are unused and rejected by decoding. Decoding does not establish legality: always use the position's legal mask and the game adapter.

## Input representation

Contiguous float32 array `[21, 8, 8]`. Row is rank minus one; column is file minus a. Consequently a1 is `[0, 0]`, regardless of who moves or how the UI is oriented. No perspective rotation is performed.

| Planes | Meaning |
| --- | --- |
| 0–5 | White pawn, knight, bishop, rook, queen, king |
| 6–11 | Black pawn, knight, bishop, rook, queen, king |
| 12 | All ones for White to move; zeros for Black |
| 13–16 | White kingside, White queenside, Black kingside, Black queenside castling rights, broadcast |
| 17 | One at the recorded en-passant target, even when no legal capture exists |
| 18 | Halfmove clock / 150, clipped to 1, broadcast |
| 19 | Current position has occurred at least twice, broadcast |
| 20 | Current position has occurred at least three times, broadcast |

All planes except the normalized clock are binary. `PLANE_NAMES` is the ordered machine-readable inventory for the later brain inspector. Repetition planes summarize history; they do not replace the full board history required by search. This compact network input can alias histories with different future draw opportunities, a documented modeling limitation.

## Outcomes and draw claims

`GameResult.white_value` is +1 White win, -1 Black win, 0 draw. `for_player(color)` converts it to the recorded position's player-to-move training target. A nonterminal position has no result (`None`), not value zero.

Automatic endings apply regardless of claim policy. `claim_draws=True` automatically accepts an available fifty-move or threefold claim, including claims available by an intended legal next move according to python-chess. This is the lightweight self-play policy. `False` allows continued play until an automatic ending; explicit human draw claims will be added in the game controller. Apply the same policy to search, masks, and result checks. Terminal masks are empty even if the rules library can still enumerate geometric legal moves in a drawn position.

Ply-limit truncation is a future self-play event, not a chess rules outcome. It must be recorded separately. Undo restores the original board history and clock through the rules library's move stack.

## Operational boundaries

Configuration validates types, ranges, unknown keys, and replay/batch compatibility before initializing storage. `checkpoint_interval` is measured in generations; `replay_buffer_size` in positions; `max_game_plies` in half-moves. The neural runtime probes usable CUDA when requested; the rules-only diagnostic does not initialize PyTorch. No model, GPU, UI, or worker is initialized at import time.

Changing action mapping or plane order requires a schema version change and explicit compatibility handling for data/checkpoints. Runtime directories are initialized under the explicitly selected workspace, never under the installed package.

Rules API reference: [python-chess core documentation](https://python-chess.readthedocs.io/en/latest/core.html).
