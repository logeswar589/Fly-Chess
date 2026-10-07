from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile

from filelock import FileLock
import numpy as np
import torch

from fly_chess.core.rules import ChessGame
from fly_chess.neural.inference import Evaluator
from fly_chess.neural.runtime import seed_everything, select_device
from fly_chess.neural.weights import load_weights
from fly_chess.search.mcts import MCTS, SearchSettings
from fly_chess.storage.checkpoints import atomic_torch_save, file_hash
from fly_chess.evaluation.statistics import summarize

# Independently sampled with replacement; the same legal move prefix is used for both colors.
OPENINGS = ((), ("e2e4", "e7e5"), ("d2d4", "d7d5"), ("c2c4", "e7e5"),
            ("g1f3", "d7d5"), ("e2e4", "c7c5"), ("d2d4", "g8f6"), ("e2e4", "e7e6"))


@dataclass(frozen=True)
class ArenaSettings:
    pairs: int = 50
    min_pairs: int = 20
    alpha: float = 0.05
    simulations: int = 32
    max_plies: int = 512
    seed: int = 42

    def __post_init__(self):
        for name in ("pairs", "min_pairs", "simulations", "max_plies"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.alpha) not in (int, float) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be uint32")


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def play_match(candidate, incumbent, *, candidate_white: bool, opening: list[str], settings: ArenaSettings,
               seed: int, cancel=None) -> dict:
    game = ChessGame(claim_draws=True)
    for move in opening:
        game.push(move)
    start_fen = game.board.fen()
    search_settings = SearchSettings(simulations=settings.simulations, noise_fraction=0)
    players = {candidate_white: MCTS(candidate, search_settings, seed=seed),
               not candidate_white: MCTS(incumbent, search_settings, seed=seed)}
    moves = []
    while True:
        result = game.result
        if result:
            score = (result.for_player(candidate_white) + 1) / 2
            return {"candidate_white": candidate_white, "opening": opening, "start_fen": start_fen,
                    "moves": moves, "seed": seed, "status": "completed", "termination": result.termination,
                    "candidate_score": score}
        if cancel is not None and cancel():
            status, termination = "aborted", "cancelled"
            break
        if len(moves) >= settings.max_plies:
            status, termination = "truncated", "ply_limit"
            break
        decision = players[game.board.turn].search(game, temperature=0, self_play=False, cancel=cancel)
        if decision.action is None:
            status, termination = "aborted", decision.stop_reason
            break
        moves.append(decision.move.uci())
        game.push_action(decision.action)
    return {"candidate_white": candidate_white, "opening": opening, "start_fen": start_fen,
            "moves": moves, "seed": seed, "status": status, "termination": termination, "candidate_score": None}


def evaluate_candidate(root: Path, candidate_path: Path, *, evaluation_id: str,
                       settings: ArenaSettings | None = None, device: str = "auto", cancel=None,
                       match_runner=play_match) -> dict:
    """Resume a fixed evaluation journal; promote only once after a complete eligible match."""
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    settings = settings or ArenaSettings()
    if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", evaluation_id) is None:
        raise ValueError("Invalid evaluation ID")
    if not candidate_path.is_file():
        raise ValueError(f"Candidate file does not exist: {candidate_path}")
    best_path = root / "models/fly_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = root / f"logs/evaluation/{evaluation_id}.json"
    candidate_hash = file_hash(candidate_path)
    with FileLock(str(root / "models/.evaluation.lock"), timeout=0):
        report = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else None
        if report and (report["candidate_sha256"] != candidate_hash or report["settings"] != asdict(settings)):
            raise ValueError("Evaluation ID already belongs to different weights or settings")
        search_contract = asdict(SearchSettings(simulations=settings.simulations, noise_fraction=0))
        if report and (report.get("evaluation_schema") != 1 or report.get("search_settings") != search_contract
                       or report.get("opening_distribution") != [list(m) for m in OPENINGS]):
            raise ValueError("Evaluation journal is incompatible with the current search/opening contract")
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
        if best_payload and best_payload.get("promotion", {}).get("evaluation_id") == evaluation_id:
            promoted = best_payload["promotion"]
            if promoted["candidate_sha256"] != candidate_hash or promoted["settings"] != asdict(settings):
                raise ValueError("Promoted evaluation has a different candidate or settings")
            atomic_json(promoted, journal_path)
            return promoted
        if report and report["status"] == "complete" and not report["promoted"]:
            return report
        seed_everything(settings.seed)
        selected_device = select_device(device)
        candidate_model, candidate_id = load_weights(candidate_path, device=selected_device.device)
        candidate = Evaluator(candidate_model, model_id=candidate_id)
        if report is None:
            report = {"evaluation_schema": 1, "evaluation_id": evaluation_id, "candidate_id": candidate_id,
                "candidate_sha256": candidate_hash, "candidate_path": str(candidate_path),
                "incumbent_sha256": file_hash(best_path) if best_path.exists() else None,
                "settings": asdict(settings), "opening_distribution": [list(m) for m in OPENINGS],
                "search_settings": search_contract, "torch_version": str(torch.__version__),
                "device": str(selected_device.device), "games": [], "status": "running", "promoted": False}
            atomic_json(report, journal_path)
        if not best_path.exists():
            if report["incumbent_sha256"] is not None:
                raise ValueError("Evaluation incumbent is missing")
            report.update(status="baseline", promoted=True, reason="Initial baseline; no improvement claim", statistics=None)
            atomic_json(report, journal_path)
            payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
            if file_hash(candidate_path) != candidate_hash:
                raise ValueError("Candidate changed before baseline creation")
            payload["promotion"] = report
            atomic_torch_save(payload, best_path)
            return report
        if report["incumbent_sha256"] != file_hash(best_path):
            raise ValueError("Incumbent changed during evaluation; use a new evaluation ID")
        incumbent_model, incumbent_id = load_weights(best_path, device=selected_device.device)
        report["incumbent_id"] = incumbent_id
        incumbent = Evaluator(incumbent_model, model_id=incumbent_id)
        rng = np.random.default_rng(settings.seed)
        opening_indices = rng.integers(0, len(OPENINGS), size=settings.pairs).tolist()
        start = len(report["games"])
        for game_index in range(start, 2 * settings.pairs):
            if cancel is not None and cancel():
                report["status"] = "interrupted"
                atomic_json(report, journal_path)
                return report
            opening = list(OPENINGS[opening_indices[game_index // 2]])
            game = match_runner(candidate, incumbent, candidate_white=game_index % 2 == 0,
                opening=opening, settings=settings, seed=(settings.seed + game_index // 2) % 2**32, cancel=cancel)
            if game["status"] == "aborted":
                report.update(status="interrupted", interrupted_game=game)
                atomic_json(report, journal_path)
                return report
            report["games"].append(game)
            report.pop("interrupted_game", None)
            atomic_json(report, journal_path)
        statistics = summarize(report["games"], requested_pairs=settings.pairs, min_pairs=settings.min_pairs, alpha=settings.alpha)
        report.update(status="complete", statistics=statistics, promoted=statistics["promote"],
                      reason="Confidence bound exceeds 50%" if statistics["promote"] else "Insufficient evidence; incumbent retained")
        atomic_json(report, journal_path)
        if report["promoted"]:
            if file_hash(best_path) != report["incumbent_sha256"] or file_hash(candidate_path) != candidate_hash:
                raise ValueError("Model files changed before promotion")
            payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
            payload["promotion"] = report
            atomic_torch_save(payload, best_path)
        return report
