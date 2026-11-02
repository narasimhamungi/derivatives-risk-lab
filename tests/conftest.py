import copy
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "book.yml"


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def small_cfg(cfg):
    """The shipped configuration with fewer paths and tree steps, for speed."""
    c = copy.deepcopy(cfg)
    c["hedge"]["paths"] = 2000
    c["greeks_check"]["mc_paths"] = 40_000
    c["greeks_check"]["tree_steps"] = 300
    return c
