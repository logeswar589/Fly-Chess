# Search contracts — Phase 3

`MCTS(evaluator, SearchSettings(...), seed=...)` owns an RNG and uses an evaluator with a frozen model. Each call builds a fresh tree; no transpositions or subtree reuse are implemented. Every simulation starts from a history-preserving `ChessGame.copy()` and follows validated legal actions. Repetition/draw-claim policy is inherited unchanged.

## Selection and backup

Selection maximizes `Q + cpuct * prior * sqrt(1 + parent_visits) / (1 + edge_visits)`. Ties prefer larger priors, then smaller action IDs. Unvisited Q is zero. Evaluations return the player-to-move value. Backup negates it once per traversed edge, so each edge's Q is for the player making that move. Opponent nodes maximize their own outcome, not the root player's outcome.

Terminal results are resolved from chess rules without a network call. The root is expanded before simulations and is not itself counted as a simulation. Each fully completed simulation increments exactly one root edge. Unfinished simulations do not update any visits. At the configured depth horizon, use the cached node evaluation; report a depth cutoff, not a draw.

Legal network priors are normalized and given a tiny exploration floor. During explicitly requested self-play only, root priors are blended with seeded Dirichlet noise (default alpha 0.3, fraction 0.25). Evaluation/human-play search gets no root noise by default. Search is inspired by [AlphaZero](https://arxiv.org/abs/1712.01815), not a reproduction of its scale or playing strength.

## Result and move sampling

`SearchResult.policy` is the full action-space normalized root visit distribution, before temperature. The selected action is argmax at temperature zero or sampled from a temperature-adjusted distribution otherwise. This separates training targets from move sampling. Terminal/cancelled searches never select a move.

`policy_source` is explicit:

- `mcts_visits`: at least one completed simulation.
- `network_fallback_no_visits`: root evaluated, but budget prevented any completed simulation.
- `network_policy`: MCTS intentionally disabled using `use_mcts=False` / `mcts_enabled=false`.
- `none`: no available policy, such as cancellation/time limit before root evaluation or a terminal root.

Do not label network-only policies as search-improved targets. A cancelled search may retain partial visit data for diagnostics, but has no selected action and should not be committed as a played move. The caller must check action/stop_reason before moving. Root `value` is mean backed-up root outcome when visits exist, otherwise the evaluated root value; it is None if no evaluation completed.

## Resource limits and cancellation

Budgets include simulations, soft wall time, allocated nodes, and search depth. Zero time stops before network inference. Node count includes the root and allocated children, even if an interruption leaves a child unexpanded. Each node stores legal edges; memory is bounded by node budget times legal branching, not just a scalar node count.

Pass a callable such as `threading.Event.is_set` for cooperative cancellation. Checks occur between traversal steps and before/after network evaluation. Python cannot interrupt an in-flight PyTorch kernel here; a time limit may overrun by a rules/evaluation operation and result construction. An interrupted evaluation's result is excluded from visit backup. Future UI work must run search outside the event loop and reject stale game/request IDs. This is not a hard real-time chess-clock guarantee.

## Brain inspection and opponent extension

Results include root candidate network/search priors, visits, parent-relative Q, root/network values, principal variation, actual simulation/node counts, elapsed time, and stop reason. Model/position/game/request IDs accompany every snapshot. An optional breadth-first tree excerpt is limited to 0–4096 edges; the default is off. Unvisited candidates have no observed Q evidence even though their displayed Q defaults to zero. The principal variation follows visits and may end at an unexpanded node; it is not a proof of best play.

The future temporary opponent model plugs into `opponent_prior(game_copy, legal_actions, baseline_probabilities)`. Its output is validated and blended only at opponent-to-root nodes, with a maximum weight of 0.25. Positive baseline exploration remains for every legal reply and adversarial backup is unchanged. This extension is forbidden in self-play and policy-only mode. The caller must provide an immutable, match-scoped profile snapshot; no learning profile is implemented in Phase 3.

## Verification limits

Fixtures check multi-ply signs, adversarial replies, mates for both colors, repetition below the root, legal normalized policies, visit conservation, noise isolation, deterministic CPU neural search, cancellation, soft-time/node limits, invalid outputs, and bounded snapshots. A real CUDA diagnostic is also run. The current model remains randomly initialized; these are correctness checks, not evidence of playing strength. Batched leaf inference and tree reuse are future measured optimizations.
