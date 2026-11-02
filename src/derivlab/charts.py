"""Charts (PNG). Validated three-slot palette, neutral text, hairline grid,
one measure per axis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, MUTED, GRID, SURFACE = "#52514e", "#8a8984", "#e6e6e3", "#ffffff"
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def _save(fig, path, synthetic):
    if synthetic:
        fig.text(0.99, 0.01, "SYNTHETIC DATA", ha="right", va="bottom", fontsize=8, color="#c00000",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def ladder(lad: pd.DataFrame, path: Path, synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    base = lad[lad["vol_shock"] == 0].sort_values("spot_shock")
    x = base["spot_shock"] * 100
    ax.plot(x, base["full_revaluation"] / 1e3, color=SERIES[0], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, label="Full revaluation")
    ax.plot(x, base["delta_gamma_vega"] / 1e3, color=SERIES[1], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, label="Delta-gamma approximation")
    for dv, color in ((lad["vol_shock"].min(), SERIES[2]),):
        g = lad[lad["vol_shock"] == dv].sort_values("spot_shock")
        ax.plot(g["spot_shock"] * 100, g["full_revaluation"] / 1e3, color=color, linewidth=2,
                label=f"Full revaluation, vol {dv * 100:+.0f} points")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Shock to every underlying (%)")
    ax.set_ylabel("Book P&L ($ thousands)")
    ax.legend(loc="lower center", fontsize=8)
    return _save(fig, path, synthetic)


def hedge_frequency(df: pd.DataFrame, path: Path) -> Path:
    """Simulated GBM paths in every mode, so no synthetic-data stamp."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(df["rebalance_days"], df["std_pct_premium"] * 100, color=SERIES[0], linewidth=2, marker="o",
            markersize=6, markeredgecolor=SURFACE, label="Simulated")
    ref = df["std_pct_premium"].iloc[0] * np.sqrt(df["rebalance_days"] / df["rebalance_days"].iloc[0]) * 100
    ax.plot(df["rebalance_days"], ref, color=SERIES[1], linewidth=2, label="Square-root-of-interval rule")
    ax.set_xticks(df["rebalance_days"])
    ax.set_xlabel("Days between rebalances (simulated paths, realised volatility = implied)")
    ax.set_ylabel("Std. dev. of hedged P&L (% of premium)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path, False)


def hedge_vol(df: pd.DataFrame, path: Path) -> Path:
    """Simulated GBM paths in every mode, so no synthetic-data stamp."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = df["realised_vol"] * 100
    ax.plot(x, df["expected_from_gamma"], color=SERIES[1], linewidth=2, zorder=2,
            label=r"$\sum\ \frac{1}{2}\Gamma S^2(\sigma_{implied}^2-\sigma_{realised}^2)\,\Delta t$")
    ax.plot(x, df["mean_pnl"], linestyle="none", marker="o", markersize=7, color=SERIES[0], markeredgecolor=SURFACE,
            zorder=3, label="Mean hedged P&L, simulated")
    ax.axvline(df["implied_vol"].iloc[0] * 100, color=MUTED, linewidth=1)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel(f"Realised volatility of simulated paths (%); implied fixed at {df['implied_vol'].iloc[0]:.0%}")
    ax.set_ylabel("Short-option hedged P&L ($ per option)")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, path, False)


def attribution(table: pd.DataFrame, path: Path, synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    cum = table[["step", "gamma", "theta", "carry"]].cumsum()
    ax.plot(cum.index, cum["step"], color="#0b0b0b", linewidth=2, label="Total hedged P&L")
    for col, color, label in (("gamma", SERIES[0], "Gamma"), ("theta", SERIES[1], "Theta"),
                              ("carry", SERIES[2], "Carry")):
        ax.plot(cum.index, cum[col], color=color, linewidth=2, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Cumulative P&L ($ per option)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    return _save(fig, path, synthetic)


def var_paths(tbl: pd.DataFrame, alpha: float, path: Path, synthetic: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    t = tbl[tbl["alpha"] == alpha]
    first = t[t["method"] == "full_revaluation_hs"]
    ax.bar(pd.to_datetime(first["date"]), first["realised_pnl"] / 1e3, width=1.5, color=MUTED, label="Realised P&L")
    for (method, label), color in zip((("full_revaluation_hs", "Full revaluation HS"), ("delta_hs", "Delta HS"),
                                       ("delta_normal", "Delta-normal")), SERIES):
        g = t[t["method"] == method]
        ax.plot(pd.to_datetime(g["date"]), -g["var"] / 1e3, color=color, linewidth=1.5, label=f"-VaR, {label}")
    ax.set_ylabel(f"Daily P&L and -VaR at {alpha:.0%} ($ thousands)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=4)
    return _save(fig, path, synthetic)
