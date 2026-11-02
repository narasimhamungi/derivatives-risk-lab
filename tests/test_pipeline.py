import json

import pytest
import yaml

from derivlab import cli, run


def test_synthetic_run_writes_everything(small_cfg, tmp_path):
    res = run.run(small_cfg, tmp_path, synthetic=True)
    s = res["summary"]
    for name in ("greeks_validation.csv", "book_sensitivities.csv", "scenario_ladder.csv", "var_backtest_daily.csv",
                 "var_backtest_summary.csv", "hedge_frequency.csv", "hedge_realised_vol.csv",
                 "hedge_example_attribution.csv", "price_validation.csv", "scenario_ladder.png", "hedge_frequency.png",
                 "hedge_realised_vol.png", "hedge_example_attribution.png", "var_backtest.png", "report.md",
                 "summary.json"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert s["synthetic"] and all(u.startswith("SYN") for u in s["data"]["underlyings"])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in report and "AAPL" not in report
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["hedge_example"]["underlying"] == "SYN1"
    assert {v["method"] for v in s["var"]} == {"full_revaluation_hs", "delta_gamma_hs", "delta_hs", "delta_normal"}
    ex = s["hedge_example"]
    assert ex["total"] == pytest.approx(ex["gamma"] + ex["theta"] + ex["carry"] + ex["residual"], abs=1e-9)


def test_cli_run_and_check(small_cfg, tmp_path, capsys):
    path = tmp_path / "book.yml"
    path.write_text(yaml.safe_dump(small_cfg), encoding="utf-8")
    assert cli.main(["--config", str(path), "check"]) == 0
    assert "gamma" in capsys.readouterr().out
    out = tmp_path / "out"
    assert cli.main(["--config", str(path), "run", "--synthetic", "--out", str(out)]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out and (out / "report.md").exists()


def test_cli_reports_errors(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "missing.yml"), "check"]) == 2
    assert "error:" in capsys.readouterr().err


def test_report_counts_read_as_english():
    import pandas as pd

    from derivlab import report
    rows = [{"method": m, "alpha": 0.99, "forecasts": 1716, "exceptions": n, "expected": 17.2, "kupiec_p": 0.1,
             "kupiec_5pct": "not rejected", "red_share": 0.0, "latest_var": 1.0, "loss_over_es": 1.2}
            for m, n in (("full_revaluation_hs", 23), ("delta_gamma_hs", 22), ("delta_hs", 31), ("delta_normal", 32))]
    text = "\n".join(report._var(pd.DataFrame(rows)))
    assert "1 fewer exception than full revaluation" in text and "1 more exception than delta HS" in text
    assert "8 more exceptions than full revaluation" in text
