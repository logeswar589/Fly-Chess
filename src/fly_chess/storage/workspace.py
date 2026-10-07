from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    def initialize(self) -> None:
        for relative in (
            "data/selfplay", "data/human_games", "data/replay_buffer", "models", "logs"
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def configure_logging(self) -> logging.Logger:
        logger = logging.getLogger("fly_chess")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # Repeated CLI invocations in the same process must not duplicate handlers.
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        handler = RotatingFileHandler(
            self.root / "logs/fly_chess.log", maxBytes=2_000_000,
            backupCount=3, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger
