#!/usr/bin/env python3
"""Generate the six project figures, degrading explicitly for missing data."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _empty(title: str, reason: str) -> plt.Figure:
    warnings.warn(f"{title}: {reason}", stacklevel=2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=15)
    ax.text(0.5, 0.42, f"Data unavailable: {reason}", ha="center", va="center", wrap=True)
    return fig


def error_vs_dim(df: pd.DataFrame) -> plt.Figure:
    needed = {"dim", "rel_error", "method", "equation"}
    if not needed <= set(df):
        return _empty("Error vs dimension", f"missing {sorted(needed - set(df))}")
    data = df[df["rel_error"].notna() & (df["rel_error"] > 0)]
    if data.empty:
        return _empty("Error vs dimension", "no positive relative-error observations")
    equations = sorted(data["equation"].unique())
    fig, axes = plt.subplots(1, len(equations), figsize=(6 * len(equations), 4), squeeze=False)
    for ax, equation in zip(axes[0], equations, strict=True):
        subset = data[data["equation"] == equation]
        for method, group in subset.groupby("method"):
            stats = group.groupby("dim")["rel_error"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                stats["dim"], stats["mean"], yerr=stats["std"].fillna(0), marker="o", label=method
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="dimension d",
            ylabel="relative error",
            title=equation,
        )
        ax.legend()
    return fig


def pareto(df: pd.DataFrame) -> plt.Figure:
    needed = {"wall_clock_sec", "rel_error", "method", "dim"}
    if not needed <= set(df):
        return _empty("Cost-error Pareto", f"missing {sorted(needed - set(df))}")
    data = df.dropna(subset=["wall_clock_sec", "rel_error"])
    data = data[(data["wall_clock_sec"] > 0) & (data["rel_error"] > 0)]
    if data.empty:
        return _empty("Cost-error Pareto", "no complete cost/error pairs")
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in data.groupby("method"):
        ax.scatter(
            group["wall_clock_sec"], group["rel_error"], c=group["dim"], label=method, alpha=0.8
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="wall-clock seconds",
        ylabel="relative error",
        title="Cost-error Pareto",
    )
    ax.legend()
    return fig


def tt_rank(df: pd.DataFrame) -> plt.Figure:
    rank_col = "tt_ranks" if "tt_ranks" in df else "max_rank"
    needed = {"dim", rank_col}
    if not needed <= set(df):
        return _empty("TT rank scaling", f"missing {sorted(needed - set(df))}")
    data = df[df[rank_col].notna()].copy()
    if data.empty:
        return _empty("TT rank scaling", "no TT rank observations")
    if rank_col == "tt_ranks":
        import json

        data["rank"] = data[rank_col].map(
            lambda value: max(json.loads(value)) if isinstance(value, str) else max(value)
        )
    else:
        data["rank"] = data[rank_col]
    fig, ax = plt.subplots(figsize=(7, 4))
    group_col = "target_error" if "target_error" in data else "equation"
    for key, group in data.groupby(group_col):
        ax.plot(group["dim"], group["rank"], marker="o", label=str(key))
    ax.set(xlabel="dimension d", ylabel="maximum TT rank", title="TT rank scaling")
    ax.legend(title=group_col)
    return fig


def precision(df: pd.DataFrame) -> plt.Figure:
    needed = {"precision", "rel_error", "throughput"}
    if not needed <= set(df):
        return _empty("Precision and throughput", f"missing {sorted(needed - set(df))}")
    data = df.dropna(subset=["rel_error", "throughput"])
    if data.empty:
        return _empty("Precision and throughput", "no precision comparison observations")
    stats = data.groupby("precision")[["rel_error", "throughput"]].mean()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stats))
    ax1.bar(x - 0.2, stats["rel_error"], width=0.4, label="relative error")
    ax1.set_yscale("log")
    ax1.set_ylabel("relative error")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, stats["throughput"], width=0.4, color="tab:orange", label="iterations/s")
    ax2.set_ylabel("iterations/s")
    ax1.set_xticks(x, stats.index)
    ax1.set_title("Precision and throughput")
    return fig


def seed_variance(df: pd.DataFrame) -> plt.Figure:
    needed = {"equation", "dim", "rel_error"}
    if not needed <= set(df):
        return _empty("Seed variance", f"missing {sorted(needed - set(df))}")
    data = df.dropna(subset=["rel_error"])
    if data.empty:
        return _empty("Seed variance", "no repeated-seed error observations")
    data = data.copy()
    data["group"] = data["equation"].astype(str) + " d=" + data["dim"].astype(str)
    groups = [group["rel_error"].to_numpy() for _, group in data.groupby("group")]
    labels = [str(key) for key, _ in data.groupby("group")]
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 0.7), 4))
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="relative error", title="Seed variance")
    return fig


def scaling(df: pd.DataFrame) -> plt.Figure:
    world_col = "hardware_world_size" if "hardware_world_size" in df else "world_size"
    needed = {world_col, "throughput"}
    if not needed <= set(df):
        return _empty("Strong scaling", f"missing {sorted(needed - set(df))}")
    data = df.dropna(subset=[world_col, "throughput"])
    if data.empty:
        return _empty("Strong scaling", "no scaling observations")
    stats = data.groupby(world_col)["throughput"].mean().sort_index()
    baseline = stats.iloc[0]
    efficiency = stats / (baseline * stats.index.to_numpy() / stats.index[0])
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(stats.index, stats.values, marker="o", label="throughput")
    ax1.set(xlabel="world size", ylabel="iterations/s", title="Strong scaling")
    ax2 = ax1.twinx()
    ax2.plot(stats.index, efficiency.values, marker="s", color="tab:orange", label="efficiency")
    ax2.set_ylabel("parallel efficiency")
    ax2.set_ylim(0, 1.1)
    return fig


FIGURES: list[tuple[str, Callable[[pd.DataFrame], plt.Figure]]] = [
    ("f1_error_vs_dim", error_vs_dim),
    ("f2_cost_error_pareto", pareto),
    ("f3_tt_rank", tt_rank),
    ("f4_precision_throughput", precision),
    ("f5_seed_variance", seed_variance),
    ("f6_strong_scaling", scaling),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figs"))
    parser.add_argument("--findings", type=Path, default=Path("docs/findings.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.summary)
    for stem, builder in FIGURES:
        _save(builder(frame), args.out, stem)
    completed = int((frame.get("status", pd.Series(dtype=str)) == "completed").sum())
    args.findings.parent.mkdir(parents=True, exist_ok=True)
    args.findings.write_text(
        "# Findings\n\n"
        f"This draft is generated from `{args.summary}`. It contains {len(frame)} runs, "
        f"of which {completed} are completed. Figures explicitly mark unavailable evidence; "
        "scientific conclusions must be added only after provenance validation.\n",
        encoding="utf-8",
    )
    print(f"figures={len(FIGURES)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
