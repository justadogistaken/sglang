"""
Analyze suffix speculative decoding stats collected by SpecStatsLogger.

Usage:
    python tools/analyze_spec_stats.py /tmp/spec_stats.jsonl
    python tools/analyze_spec_stats.py /tmp/spec_stats.jsonl --output-dir /tmp/spec_plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # usable_drafts = draft_len - 1 (exclude root)
    df["usable_drafts"] = df["draft_len"] - 1
    return df


def percentile_table(series: pd.Series, name: str) -> str:
    pcts = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(series.dropna(), pcts)
    lines = [f"  {name}"]
    lines += [f"    P{p:3d}: {v:.4f}" for p, v in zip(pcts, vals)]
    return "\n".join(lines)


def pearson(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    mask = a.notna() & b.notna()
    r, p = scipy_stats.pearsonr(a[mask], b[mask])
    return r, p


def find_threshold(df: pd.DataFrame, col: str, target_accept_rate: float = 0.05,
                   n_bins: int = 20) -> float:
    """Find the value of `col` below which mean accept_rate < target_accept_rate."""
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    bins = np.linspace(lo, hi, n_bins + 1)
    df2 = df.copy()
    df2["_bin"] = pd.cut(df[col], bins)
    grouped = df2.groupby("_bin", observed=True)["accept_rate"].mean()
    for interval, mean_ar in grouped.items():
        if mean_ar >= target_accept_rate:
            return interval.left
    return hi


# ── analysis sections ─────────────────────────────────────────────────────────

def section_overview(df: pd.DataFrame) -> str:
    n = len(df)
    n_steps = df["step"].nunique()
    n_reqs  = df["req_id"].nunique()
    mean_ar = df["accept_rate"].mean()
    zero_ar = (df["accept_rate"] == 0).mean()
    mean_dl = df["draft_len"].mean()
    mean_al = df["accept_len"].mean()

    return f"""
═══════════════════════════════════════════════════
 OVERVIEW
═══════════════════════════════════════════════════
  Total records      : {n:,}
  Unique steps       : {n_steps:,}
  Unique requests    : {n_reqs:,}

  mean accept_rate   : {mean_ar:.4f}  ({mean_ar*100:.1f}%)
  zero accept_rate % : {zero_ar*100:.1f}%   (steps where nothing was accepted)
  mean draft_len     : {mean_dl:.2f}
  mean accept_len    : {mean_al:.2f}
"""


def section_distributions(df: pd.DataFrame) -> str:
    lines = ["""
═══════════════════════════════════════════════════
 DISTRIBUTIONS
═══════════════════════════════════════════════════"""]
    for col in ["first_token_prob", "draft_score_avg", "match_len", "draft_len", "accept_rate"]:
        if col in df.columns:
            lines.append(percentile_table(df[col], col))
    return "\n".join(lines)


def section_correlations(df: pd.DataFrame) -> str:
    predictors = ["first_token_prob", "draft_score_avg", "match_len", "draft_len"]
    lines = ["""
═══════════════════════════════════════════════════
 CORRELATIONS WITH accept_rate
═══════════════════════════════════════════════════
  (Pearson r,  p-value)"""]
    for col in predictors:
        if col not in df.columns:
            continue
        r, p = pearson(df[col], df["accept_rate"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        lines.append(f"  {col:<22}: r={r:+.4f}  p={p:.2e}  {sig}")
    return "\n".join(lines)


def section_thresholds(df: pd.DataFrame) -> str:
    lines = ["""
═══════════════════════════════════════════════════
 THRESHOLD ANALYSIS
═══════════════════════════════════════════════════
  Below what value does mean accept_rate drop under 5%?
"""]

    for col in ["first_token_prob", "draft_score_avg", "match_len"]:
        if col not in df.columns:
            continue
        thr = find_threshold(df, col, target_accept_rate=0.05)
        below = (df[col] < thr).mean() * 100
        lines.append(f"  {col:<22}: threshold ≈ {thr:.3f}  "
                     f"({below:.1f}% of records would be skipped)")

    return "\n".join(lines)


def section_prob_decay(df: pd.DataFrame) -> str:
    """Analyze how probs decay along the draft path."""
    if "probs" not in df.columns:
        return ""

    # Expand probs list into columns
    max_pos = 8
    prob_cols = []
    for i in range(max_pos):
        col = f"prob_pos{i+1}"
        df[col] = df["probs"].apply(
            lambda p: p[i] if isinstance(p, list) and len(p) > i else np.nan
        )
        prob_cols.append(col)

    lines = ["""
═══════════════════════════════════════════════════
 PROB DECAY ALONG DRAFT PATH
═══════════════════════════════════════════════════
  Mean cache probability at each draft position:
"""]
    for i, col in enumerate(prob_cols):
        mean_p = df[col].mean()
        if np.isnan(mean_p):
            break
        bar = "█" * int(mean_p * 30)
        lines.append(f"  pos {i+1:2d}: {mean_p:.4f}  {bar}")

    # Breakeven: at what position does mean prob typically drop below 0.3?
    means = [df[c].mean() for c in prob_cols]
    breakeven = next((i+1 for i, m in enumerate(means) if not np.isnan(m) and m < 0.3), None)
    if breakeven:
        lines.append(f"\n  Mean prob drops below 0.30 at position {breakeven}")
        lines.append(f"  → Suggests effective draft_token_num ≈ {breakeven}")

    return "\n".join(lines)


def section_binned_analysis(df: pd.DataFrame) -> str:
    """For each first_token_prob bucket, show mean accept_rate."""
    if "first_token_prob" not in df.columns:
        return ""

    lines = ["""
═══════════════════════════════════════════════════
 ACCEPT RATE BY first_token_prob BUCKET
═══════════════════════════════════════════════════
  bucket            count   mean_accept_rate  zero_rate
"""]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5",
              "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]
    df2 = df.copy()
    df2["ftp_bin"] = pd.cut(df["first_token_prob"], bins=bins, labels=labels, right=False)
    for label, grp in df2.groupby("ftp_bin", observed=True):
        if len(grp) == 0:
            continue
        mean_ar = grp["accept_rate"].mean()
        zero_r  = (grp["accept_rate"] == 0).mean()
        bar = "▓" * int(mean_ar * 20)
        lines.append(f"  [{label}]  n={len(grp):6d}  "
                     f"accept={mean_ar:.3f} {bar:<20}  zero={zero_r:.1%}")

    return "\n".join(lines)


def section_draft_len_efficiency(df: pd.DataFrame) -> str:
    """Does longer draft_len actually help?"""
    lines = ["""
═══════════════════════════════════════════════════
 DRAFT LENGTH EFFICIENCY
═══════════════════════════════════════════════════
  draft_len  count   mean_accept_rate  mean_accept_len  efficiency
  (efficiency = accept_len / usable_drafts)
"""]
    for dl, grp in df.groupby("draft_len"):
        if len(grp) < 10:
            continue
        mar = grp["accept_rate"].mean()
        mal = grp["accept_len"].mean()
        eff = (grp["accept_len"] / grp["usable_drafts"].clip(lower=1)).mean()
        lines.append(f"  dl={dl:3d}  n={len(grp):6d}  "
                     f"accept_rate={mar:.3f}  accept_len={mal:.2f}  eff={eff:.3f}")

    return "\n".join(lines)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand probs if present
    max_pos = 8
    if "probs" in df.columns:
        for i in range(max_pos):
            col = f"prob_pos{i+1}"
            if col not in df.columns:
                df[col] = df["probs"].apply(
                    lambda p: p[i] if isinstance(p, list) and len(p) > i else np.nan
                )

    # ── Figure 1: correlation overview ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Suffix Speculative Decoding — Draft Quality vs Accept Rate", fontsize=14)

    predictors = [
        ("first_token_prob", "First Token Prob (cache freq)"),
        ("draft_score_avg",  "Draft Score Avg (mean prob)"),
        ("match_len",        "Match Length"),
        ("draft_len",        "Draft Length"),
    ]
    for ax, (col, label) in zip(axes.flat, predictors):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        # Hexbin for dense data
        hb = ax.hexbin(df[col], df["accept_rate"], gridsize=40, cmap="Blues",
                       mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="count")
        r, p = pearson(df[col], df["accept_rate"])
        ax.set_xlabel(label)
        ax.set_ylabel("accept_rate")
        ax.set_title(f"r = {r:+.3f}  (p={p:.2e})")
        # Overlay binned mean
        bins = np.linspace(df[col].quantile(0.01), df[col].quantile(0.99), 15)
        df["_tmp"] = pd.cut(df[col], bins)
        means = df.groupby("_tmp", observed=True)["accept_rate"].mean()
        mids  = [(iv.left + iv.right) / 2 for iv in means.index]
        ax.plot(mids, means.values, "r-o", ms=4, lw=1.5, label="bin mean")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path1 = output_dir / "1_correlations.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  saved: {path1}")

    # ── Figure 2: first_token_prob threshold analysis ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("First Token Prob — Threshold Analysis", fontsize=13)

    bins = np.linspace(0, 1, 21)
    df["ftp_bin"] = pd.cut(df["first_token_prob"], bins)
    grp = df.groupby("ftp_bin", observed=True)
    mids = [(iv.left + iv.right) / 2 for iv in grp["accept_rate"].mean().index]
    mean_ar = grp["accept_rate"].mean().values
    zero_r  = (grp["accept_rate"] == 0).mean().values
    counts  = grp["accept_rate"].count().values

    ax = axes[0]
    ax.bar(mids, mean_ar, width=0.045, alpha=0.7, color="steelblue", label="mean accept_rate")
    ax.set_xlabel("first_token_prob")
    ax.set_ylabel("mean accept_rate")
    ax.set_title("Mean Accept Rate per first_token_prob bucket")
    ax.axhline(0.05, color="red", ls="--", lw=1, label="5% threshold")
    ax.legend()

    ax2 = axes[0].twinx()
    ax2.plot(mids, counts, "g--o", ms=3, lw=1, alpha=0.6, label="count")
    ax2.set_ylabel("count", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    ax = axes[1]
    ax.bar(mids, zero_r, width=0.045, alpha=0.7, color="tomato", label="zero accept_rate %")
    ax.set_xlabel("first_token_prob")
    ax.set_ylabel("fraction with accept_rate = 0")
    ax.set_title("Fraction of Fully-Rejected Steps per first_token_prob bucket")
    ax.legend()

    plt.tight_layout()
    path2 = output_dir / "2_first_token_threshold.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  saved: {path2}")

    # ── Figure 3: prob decay along draft path ─────────────────────────────────
    if "prob_pos1" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Prob Decay Along Draft Path", fontsize=13)

        positions = list(range(1, max_pos + 1))
        prob_cols = [f"prob_pos{i}" for i in positions]

        # Overall mean decay
        means = [df[c].mean() for c in prob_cols if c in df.columns]
        stds  = [df[c].std()  for c in prob_cols if c in df.columns]
        pos_x = positions[:len(means)]

        ax = axes[0]
        ax.errorbar(pos_x, means, yerr=stds, marker="o", capsize=4,
                    color="steelblue", lw=2, label="mean ± std")
        ax.axhline(0.3, color="red",    ls="--", lw=1, label="0.3 (suggested cutoff)")
        ax.axhline(0.1, color="orange", ls="--", lw=1, label="0.1 (min_token_prob)")
        ax.set_xlabel("Draft position")
        ax.set_ylabel("Cache probability")
        ax.set_title("Mean Cache Prob at Each Draft Position")
        ax.set_xticks(pos_x)
        ax.legend()

        # Decay split by accept / reject on first token
        accepted = df[df["accept_len"] > 0]
        rejected = df[df["accept_len"] == 0]
        ax = axes[1]
        for subset, label, color in [(accepted, "accept_len > 0", "steelblue"),
                                      (rejected, "accept_len = 0", "tomato")]:
            m = [subset[c].mean() for c in prob_cols if c in subset.columns]
            ax.plot(pos_x[:len(m)], m, marker="o", lw=2, color=color, label=label)
        ax.set_xlabel("Draft position")
        ax.set_ylabel("Cache probability")
        ax.set_title("Prob Decay: Accepted vs Rejected Steps")
        ax.set_xticks(pos_x)
        ax.legend()

        plt.tight_layout()
        path3 = output_dir / "3_prob_decay.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        print(f"  saved: {path3}")

    # ── Figure 4: match_len analysis ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Match Length Analysis", fontsize=13)

    ml_bins = np.arange(0, df["match_len"].quantile(0.99) + 2, 1)
    df["ml_bin"] = pd.cut(df["match_len"], bins=ml_bins)
    grp = df.groupby("ml_bin", observed=True)
    ml_mids   = [(iv.left + iv.right) / 2 for iv in grp["accept_rate"].mean().index]
    ml_mean   = grp["accept_rate"].mean().values
    ml_counts = grp["accept_rate"].count().values

    ax = axes[0]
    ax.bar(ml_mids, ml_mean, width=0.8, alpha=0.7, color="mediumseagreen")
    ax.set_xlabel("match_len")
    ax.set_ylabel("mean accept_rate")
    ax.set_title("Mean Accept Rate vs Match Length")

    ax2 = axes[0].twinx()
    ax2.plot(ml_mids, ml_counts, "r--o", ms=3, lw=1, alpha=0.6)
    ax2.set_ylabel("count", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # joint: match_len + first_token_prob -> accept_rate
    ax = axes[1]
    if "first_token_prob" in df.columns:
        pivot = df.copy()
        pivot["ftp_q"] = pd.qcut(df["first_token_prob"], q=4,
                                  labels=["Q1 low","Q2","Q3","Q4 high"])
        for q_label, grp2 in pivot.groupby("ftp_q", observed=True):
            ml_grp = grp2.groupby("ml_bin", observed=True)["accept_rate"].mean()
            mids2  = [(iv.left + iv.right) / 2 for iv in ml_grp.index]
            ax.plot(mids2, ml_grp.values, marker="o", ms=3, lw=1.5, label=q_label)
        ax.set_xlabel("match_len")
        ax.set_ylabel("mean accept_rate")
        ax.set_title("Accept Rate vs Match Length (by first_token_prob quartile)")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    path4 = output_dir / "4_match_len.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  saved: {path4}")

    # ── Figure 5: skip simulation ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Skip Simulation — Tradeoff: skip rate vs wasted accept rate", fontsize=13)

    thresholds = np.linspace(0, df["first_token_prob"].quantile(0.95), 60)
    skip_rates, wasted_ars = [], []
    total = len(df)
    for thr in thresholds:
        skip_mask = df["first_token_prob"] < thr
        skip_rate = skip_mask.mean()
        # "wasted" = mean accept_rate of records we'd skip (i.e. we'd have accepted these)
        wasted_ar = df.loc[skip_mask, "accept_rate"].mean() if skip_mask.any() else 0.0
        skip_rates.append(skip_rate)
        wasted_ars.append(wasted_ar if not np.isnan(wasted_ar) else 0.0)

    ax.plot(thresholds, skip_rates,  lw=2, color="steelblue", label="skip rate (fraction of steps skipped)")
    ax.plot(thresholds, wasted_ars,  lw=2, color="tomato",    label="avg accept_rate of skipped steps")
    ax.axvline(0.3, color="gray", ls="--", lw=1, label="example threshold=0.3")
    ax.set_xlabel("first_token_prob skip threshold")
    ax.set_ylabel("rate")
    ax.set_title("If we skip spec when first_token_prob < threshold...")
    ax.legend()
    ax.set_xlim(0, thresholds[-1])
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path5 = output_dir / "5_skip_simulation.png"
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    print(f"  saved: {path5}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to spec_stats.jsonl")
    parser.add_argument("--output-dir", default="spec_analysis",
                        help="Directory for output plots (default: ./spec_analysis)")
    args = parser.parse_args()

    print(f"\nLoading {args.jsonl} ...")
    df = load_jsonl(args.jsonl)
    print(f"Loaded {len(df):,} records.")

    output_dir = Path(args.output_dir)

    # ── text report ───────────────────────────────────────────────────────────
    report = "\n".join([
        section_overview(df),
        section_distributions(df),
        section_correlations(df),
        section_thresholds(df),
        section_prob_decay(df),
        section_binned_analysis(df),
        section_draft_len_efficiency(df),
    ])
    print(report)

    report_path = output_dir / "report.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\n  report saved: {report_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_all(df, output_dir)
    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
