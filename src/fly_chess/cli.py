import argparse
import json
from pathlib import Path

from fly_chess import __version__
from fly_chess.config import ConfigurationError, load_config
from fly_chess.core.actions import ACTION_SIZE, decode_action, legal_mask
from fly_chess.core.encoding import ENCODING_SCHEMA_VERSION, encode_board
from fly_chess.core.rules import ChessGame
from fly_chess.storage.workspace import Workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fly-Chess: trainable chess, built in phases")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="TOML settings file")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Data/model/log directory root")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="Open the monochrome chess and brain interface")
    gui.add_argument("--weights", type=Path, help="Saved inference or training checkpoint for play")
    human = commands.add_parser('train-human', help='Explicitly train a candidate from consented complete human PGNs, then evaluate')
    human.add_argument('--base', type=Path, required=True)
    human.add_argument('--epochs', type=int, default=1)
    human.add_argument('--max-positions', type=int, default=4096)
    human.add_argument('--learning-rate', type=float, default=.0001)
    diagnostic = commands.add_parser("diagnose", help="Check chess rules and representation without a GUI")
    diagnostic.add_argument("--fen", help="Optional standard-chess FEN; repetition history is unavailable")
    neural = commands.add_parser("neural-diagnose", help="Run policy/value inference (untrained unless weights supplied)")
    neural.add_argument("--fen", help="Optional standard-chess FEN")
    neural.add_argument("--weights", type=Path, help="Load versioned inference weights")
    neural.add_argument("--save-weights", type=Path, help="Save inference weights, not a training checkpoint")
    neural.add_argument("--inspect", action="store_true", help="Include bounded real activation samples")
    search = commands.add_parser("search-diagnose", help="Run bounded neural MCTS; default weights are untrained")
    search.add_argument("--fen", help="Optional standard-chess FEN")
    search.add_argument("--weights", type=Path)
    search.add_argument("--seconds", type=float, help="Soft time budget; an in-flight evaluation can finish")
    search.add_argument("--self-play", action="store_true", help="Enable root exploration noise")
    search.add_argument("--inspect", action="store_true", help="Include up to 128 tree edges")
    selfplay = commands.add_parser("selfplay", help="Generate a finite batch of games and persist replay without optimization")
    selfplay.add_argument("--fen", help="Optional initial position for diagnostics")
    selfplay.add_argument("--weights", type=Path)
    train = commands.add_parser("train", help="Train a finite number of generations; checkpoints save each boundary")
    train.add_argument("--generations", type=int, default=1)
    train.add_argument("--resume", type=Path, help="Full training checkpoint; saved configuration is restored")
    arena = commands.add_parser("evaluate", help="Evaluate saved candidate against best, or establish initial baseline")
    arena.add_argument("--candidate", type=Path, required=True)
    arena.add_argument("--id", help="Stable evaluation ID to resume; omit to create a new evaluation")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        fen = getattr(args, "fen", None)
        game = ChessGame.from_fen(fen, claim_draws=config.claim_draws) if fen else ChessGame(claim_draws=config.claim_draws)
        workspace = Workspace(args.workspace.resolve())
        workspace.initialize()
        logger = workspace.configure_logging()
        if args.command == 'train-human':
            from fly_chess.training.human import train_human_candidate
            report = train_human_candidate(workspace.root, args.base, config, epochs=args.epochs,
                max_positions=args.max_positions, learning_rate=args.learning_rate)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "gui":
            from fly_chess.ui.app import run
            run(workspace.root, args.weights, config)
            return 0
        if args.command == "evaluate":
            from uuid import uuid4
            from fly_chess.evaluation.arena import ArenaSettings, evaluate_candidate
            report = evaluate_candidate(workspace.root, args.candidate, evaluation_id=args.id or uuid4().hex,
                device=config.device, settings=ArenaSettings(pairs=config.evaluation_pairs, min_pairs=config.evaluation_min_pairs,
                    simulations=config.evaluation_simulations, max_plies=config.evaluation_max_plies, seed=config.seed))
            logger.info("Evaluation: %s", report["status"])
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train":
            from fly_chess.training.controller import TrainingController
            with TrainingController(workspace.root, config if args.config or not args.resume else None, resume=args.resume) as controller:
                report = controller.run(args.generations)
                logger.info("Training complete: %s", report["progress"])
                print(json.dumps(report, indent=2))
            return 0
        if args.command == "selfplay":
            from fly_chess.selfplay.workers import run_batch
            report = run_batch(config, workspace.root, weights=args.weights, start_fen=args.fen,
                on_game=lambda progress: logger.info("Self-play committed: %s", progress))
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "search-diagnose":
            from fly_chess.search.diagnostic import diagnose_search
            report = diagnose_search(config, game, weights=args.weights, seconds=args.seconds,
                                     self_play=args.self_play, inspect=args.inspect)
            logger.info("Search diagnostic: %s", report["stop_reason"])
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "neural-diagnose":
            from fly_chess.neural.diagnostic import diagnose_network
            report = diagnose_network(config, game, weights=args.weights,
                                      save=args.save_weights, inspect=args.inspect)
            logger.info("Neural diagnostic passed on %s", report["device"])
            print(json.dumps(report, indent=2))
            return 0
        state = encode_board(game.board)
        actions = game.actions()
        mask = legal_mask(game.board, claim_draws=config.claim_draws)
        result = game.result
        report = {
            "version": __version__, "phase": 1, "status": "ok",
            "fen": game.board.fen(), "encoding_schema": ENCODING_SCHEMA_VERSION,
            "input_shape": list(state.shape), "input_dtype": str(state.dtype),
            "action_space": ACTION_SIZE, "legal_action_count": len(actions),
            "mask_count": int(mask.sum()),
            "legal_moves": [decode_action(action).uci() for action in actions],
            "result": None if result is None else {"white_value": result.white_value, "termination": result.termination},
            "claim_draws": config.claim_draws, "requested_device": config.device,
            "device_status": "Use neural-diagnose to probe CPU/CUDA and run the network",
            "workspace": str(workspace.root),
        }
        logger.info("Diagnostic passed: %s legal actions", len(actions))
        print(json.dumps(report, indent=2))
        return 0
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Training can resume from its last durable checkpoint.\n")
    except (ConfigurationError, ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"fly-chess: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
