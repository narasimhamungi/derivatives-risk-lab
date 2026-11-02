"""Command line: derivlab <command>.

  check    price the configured validation cases with all three pricers and
           report the largest difference from the closed form
  run      validate, measure the book, run the hedging studies and the options
           VaR backtest; write outputs/ and report.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from derivlab import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="derivlab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(settings.CONFIG_DIR / "book.yml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("run")
    p.add_argument("--synthetic", action="store_true",
                   help="invented returns and renamed underlyings; no database needed")
    p.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        cfg = settings.load(Path(args.config).name, Path(args.config).parent)
        if args.cmd == "check":
            from derivlab import validation
            errors = validation.error_summary(validation.greeks_table(cfg))
            print("largest relative difference from the closed form")
            print(errors.to_string(index=False, float_format=lambda x: f"{x:.2e}", na_rep="n.a."))
            return 0
        from derivlab import run as run_mod
        synthetic = args.synthetic or cfg["data"]["source"] == "synthetic"
        out = Path(args.out) if args.out else settings.OUTPUT_DIR / ("synthetic" if synthetic else "")
        res = run_mod.run(cfg, out, synthetic=synthetic)
        s = res["summary"]
        top = [v for v in s["var"] if v["alpha"] == max(x["alpha"] for x in s["var"])]
        print(f"{s['data']['returns']:,} returns, {s['data']['from']} to {s['data']['to']}"
              + (" (SYNTHETIC)" if synthetic else ""))
        if s["data"].get("issues"):
            print("price validation: " + ", ".join(f"{k} {n}" for k, n in sorted(s["data"]["issues"].items()))
                  + f" (each listed in {out / 'price_validation.csv'})")
        for v in top:
            print(f"  {v['method']:<20} {v['exceptions']:>3} exceptions vs {v['expected']:.1f} expected, "
                  f"Kupiec {v['kupiec_5pct']}")
        print(f"-> {out / 'report.md'}")
        return 0
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("DERIVLAB_DEBUG"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
