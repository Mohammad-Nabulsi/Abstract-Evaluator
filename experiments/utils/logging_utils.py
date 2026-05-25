from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Path,
    log_file: str,
    level: int = logging.INFO,
    add_stream_handler: bool = True,
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear handlers to avoid duplicate lines across notebook reruns.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def log_df_overview(logger: logging.Logger, name: str, rows: int, cols: int) -> None:
    logger.info("%s shape: (%d, %d)", name, rows, cols)


def maybe_log_score_dist(logger: Optional[logging.Logger], name: str, dist: dict) -> None:
    if logger is None:
        return
    logger.info("%s score distribution: %s", name, dist)

