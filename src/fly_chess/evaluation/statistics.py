import math


def summarize(games: list[dict], *, requested_pairs: int, min_pairs: int, alpha: float) -> dict:
    """Treat each color-swapped pair as one bounded observation, not two IID games."""
    if requested_pairs < 1 or min_pairs < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid evaluation sample/confidence settings")
    counts = {name: 0 for name in ("wins", "losses", "draws", "truncated", "aborted")}
    for game in games:
        if game["status"] == "completed":
            score = game["candidate_score"]
            if score not in (0, 0.5, 1):
                raise ValueError("Invalid evaluation score")
            counts[{0: "losses", 0.5: "draws", 1: "wins"}[score]] += 1
        elif game["status"] in ("truncated", "aborted"):
            counts[game["status"]] += 1
        else:
            raise ValueError("Unknown evaluation outcome")
    pairs = []
    for index in range(0, len(games) - 1, 2):
        first, second = games[index:index + 2]
        if first["candidate_white"] is not True or second["candidate_white"] is not False:
            raise ValueError("Evaluation colors are not paired")
        if first["status"] == second["status"] == "completed":
            pairs.append((first["candidate_score"] + second["candidate_score"]) / 2)
    complete = len(games) == 2 * requested_pairs and len(pairs) == requested_pairs
    score = sum(pairs) / len(pairs) if pairs else None
    # Distribution-free two-sided Hoeffding bound over independent sampled opening pairs.
    radius = math.sqrt(math.log(2 / alpha) / (2 * len(pairs))) if pairs else None
    interval = None if score is None else [max(0., score - radius), min(1., score + radius)]
    eligible = complete and len(pairs) >= min_pairs
    promote = bool(eligible and interval[0] > 0.5)
    elo_delta = None
    if eligible and 0 < score < 1:
        elo_delta = 400 * math.log10(score / (1 - score))
    def elo_bound(probability):
        return None if probability <= 0 or probability >= 1 else 400 * math.log10(probability / (1 - probability))
    return {**counts, "games_played": len(games), "completed_pairs": len(pairs),
        "average_game_length": sum(len(game.get("moves", [])) for game in games) / len(games) if games else None,
        "requested_pairs": requested_pairs, "score_rate": score,
        "candidate_win_rate": counts["wins"] / (counts["wins"] + counts["losses"] + counts["draws"])
            if counts["wins"] + counts["losses"] + counts["draws"] else None,
        "score_interval": interval, "confidence": 1 - alpha,
        "interval_method": "Hoeffding over sampled color-swapped pairs",
        "promotion_eligible": eligible, "promote": promote,
        "relative_elo_difference": elo_delta,
        "relative_elo_interval": [elo_bound(p) for p in interval] if eligible else None,
        "rating_label": "Relative estimate versus this opponent under these search/opening settings; not human Elo",
        "rating_status": "estimated" if elo_delta is not None else "unrated",
        "complete": complete}
