"""Paths and YAML settings."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("DERIVLAB_CONFIG_DIR", PROJECT_ROOT / "config"))
OUTPUT_DIR = Path(os.environ.get("DERIVLAB_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


def load(name: str = "book.yml", config_dir: Path = CONFIG_DIR) -> dict:
    return yaml.safe_load((Path(config_dir) / name).read_text(encoding="utf-8"))
