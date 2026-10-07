from dataclasses import asdict
from typing import Callable

import numpy as np

from fly_chess.core.encoding import encode_board
from fly_chess.core.rules import ChessGame
from fly_chess.selfplay.records import GameRecord, Sample


def play_game(search, *, game_id: str, seed: int, max_plies: int = 512,
              start: ChessGame | None = None, temperature: float = 1.0,
              temperature_plies: int = 30, cancel: Callable[[], bool] | None = None) -> GameRecord:
    if type(max_plies) is not int or max_plies < 1 or type(temperature_plies) is not int or temperature_plies < 0:
        raise ValueError("max_plies must be positive and temperature_plies nonnegative")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    game = start.copy() if start is not None else ChessGame(claim_draws=True)
    record = GameRecord(game_id, seed, search.evaluator.model_id, game.board.root().fen(),
                        [m.uci() for m in game.board.move_stack], game.claim_draws,
                        settings={"search": asdict(search.settings), "max_plies": max_plies,
                                  "temperature": temperature, "temperature_plies": temperature_plies})
    while True:
        result = game.result
        if result is not None:
            record.status, record.termination, record.white_result = "completed", result.termination, result.white_value
            break
        if cancel is not None and cancel():
            break
        if len(record.samples) >= max_plies:
            record.status, record.termination, record.white_result = "truncated", "ply_limit", 0.0
            break
        result = search.search(game, self_play=True, temperature=temperature if len(record.samples) < temperature_plies else 0,
                               cancel=cancel, game_id=game_id, request_id=f"{game_id}:{len(record.samples)}")
        if result.action is None or result.stop_reason == "cancelled":
            record.termination = "cancelled" if result.stop_reason == "cancelled" else "search_interrupted"
            break
        actions = game.actions()
        record.samples.append(Sample(encode_board(game.board), result.action, actions,
            result.policy[actions].copy(), bool(game.board.turn), float(result.snapshot["network_value"]), result.policy_source))
        game.push_action(result.action)
    for sample in record.samples:
        sample.target = None if record.white_result is None else (record.white_result if sample.white_to_move else -record.white_result)
    record.validate()
    return record
