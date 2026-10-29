from __future__ import annotations
import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str, level: str = "INFO") -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path / "ai_daily_agent.log", encoding="utf-8"),
        ],
        force=True,
    )