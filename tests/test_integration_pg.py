"""The lakehouse path end to end: market-risk-lab's price pull and validation
against a real Postgres holding the gold schema, then the full derivlab run.
The prices are seeded synthetic test input; nothing here is a result.

Needs a database whose name ends in _test (set MDL_DB_NAME, or MRL_DSN); the
fixture truncates the gold tables. Skipped when none is reachable, unless
DERIVLAB_REQUIRE_DB=1 (CI sets it).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from market_risk_lab import store
from psycopg2.extras import execute_values

from derivlab import run

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_conn():
    required = os.environ.get("DERIVLAB_REQUIRE_DB") == "1"
    try:
        conn = store.connect()
    except Exception as e:  # noqa: BLE001
        if required:
            raise
        pytest.skip(f"no Postgres available: {e}")
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        name = cur.fetchone()[0]
    if not name.endswith("_test"):
        conn.close()
        if required:
            raise RuntimeError(f"refusing to run integration tests against {name!r}")
        pytest.skip(f"database {name!r} is not a *_test database")
    yield conn
    conn.close()


@pytest.fixture
def gold(pg_conn, cfg):
    tickers = sorted({p["underlying"] for p in cfg["book"]})
    dates = pd.bdate_range("2019-01-02", "2024-06-28")
    rng = np.random.default_rng(3)
    px = 100 * np.cumprod(1 + 0.015 * rng.standard_t(5, (len(dates), len(tickers))) * np.sqrt(3 / 5), axis=0)
    with pg_conn.cursor() as cur:
        cur.execute((Path(__file__).parent / "gold_schema_fixture.sql").read_text(encoding="utf-8"))
        cur.execute("TRUNCATE dim_security, dim_date RESTART IDENTITY CASCADE")
        execute_values(cur, "INSERT INTO dim_date VALUES %s", [
            (int(d.strftime("%Y%m%d")), d.date(), d.year, d.quarter, d.month, d.month_name(), d.day,
             d.day_of_week, d.day_name(), True, True) for d in dates])
        execute_values(cur, "INSERT INTO dim_security (ticker, effective_from) VALUES %s",
                       [(t, dates[0].date()) for t in tickers])
        cur.execute("SELECT ticker, security_key FROM dim_security")
        key = dict(cur.fetchall())
        execute_values(cur, "INSERT INTO fact_price_daily_consensus VALUES %s", [
            (key[t], int(d.strftime("%Y%m%d")), round(float(px[i, j]), 6), "yfinance", ["yfinance"], None,
             "single_source") for i, d in enumerate(dates) for j, t in enumerate(tickers)], page_size=5000)
    pg_conn.commit()
    return {"tickers": tickers, "n_dates": len(dates)}


def test_lakehouse_run_end_to_end(gold, small_cfg, tmp_path):
    res = run.run(small_cfg, tmp_path, synthetic=False)
    s = res["summary"]
    assert not s["synthetic"] and "market-risk-lab" in s["data"]["source"]
    assert s["data"]["returns"] == gold["n_dates"] - 1
    assert sorted(s["data"]["underlyings"]) == gold["tickers"]
    assert len(s["var"]) == 4 * len(small_cfg["var"]["alphas"])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" not in report and "AAPL" in report
    assert s["hedge_example"]["underlying"] == small_cfg["hedge"]["example_underlying"]
