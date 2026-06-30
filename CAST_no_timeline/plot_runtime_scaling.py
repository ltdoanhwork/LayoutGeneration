#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


ROOT = Path("/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/layout/Eval_loss")
OUT_DIR = Path("/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/layout/runtime_scaling")
PNG_PATH = OUT_DIR / "runtime_scaling_area_chart.png"
SUMMARY_CSV = OUT_DIR / "runtime_scaling_summary.csv"

COUNT_RE = re.compile(r"_(\d+)_")
TIME_RE = re.compile(r"Total time:\s*([0-9.]+)s")


def parse_run(run_dir: Path):
    log_path = run_dir / "run_eval.log"
    if not log_path.is_file():
        return None

    count_match = COUNT_RE.search(run_dir.name)
    if not count_match:
        return None
    image_count = int(count_match.group(1))

    total_time = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        time_match = TIME_RE.search(line)
        if time_match:
            total_time = float(time_match.group(1))
            break
    if total_time is None:
        return None

    return {
        "run_name": run_dir.name,
        "layout": run_dir.name.split("_")[-1],
        "image_count": image_count,
        "time_sec": total_time,
    }


def collect_rows():
    rows = []
    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        row = parse_run(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["image_count"]].append(row["time_sec"])

    summary = []
    for image_count in sorted(buckets):
        vals = buckets[image_count]
        summary.append(
            {
                "image_count": image_count,
                "num_runs": len(vals),
                "mean_time_sec": mean(vals),
                "std_time_sec": pstdev(vals) if len(vals) > 1 else 0.0,
                "min_time_sec": min(vals),
                "max_time_sec": max(vals),
            }
        )
    return summary


def save_summary_csv(summary):
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_count",
                "num_runs",
                "mean_time_sec",
                "std_time_sec",
                "min_time_sec",
                "max_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def plot(summary):
    xs = [row["image_count"] for row in summary]
    ys = [row["mean_time_sec"] for row in summary]
    ymins = [row["min_time_sec"] for row in summary]
    ymaxs = [row["max_time_sec"] for row in summary]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)

    fig.patch.set_facecolor("#f8f4ec")
    ax.set_facecolor("#fffaf2")

    ax.fill_between(
        xs,
        ymins,
        ymaxs,
        color="#f2a65a",
        alpha=0.28,
        label="Min-max range across layouts",
        zorder=1,
    )
    ax.fill_between(
        xs,
        ys,
        color="#d95d39",
        alpha=0.12,
        zorder=0,
    )
    ax.plot(
        xs,
        ys,
        color="#b23a24",
        linewidth=3.2,
        marker="o",
        markersize=8,
        markerfacecolor="#fff2cc",
        markeredgecolor="#b23a24",
        markeredgewidth=2.0,
        label="Mean runtime",
        zorder=3,
    )

    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.1f}s",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="#6b2418",
            fontweight="bold",
        )

    ax.set_title(
        "Runtime Scaling with Number of Images",
        fontsize=20,
        fontweight="bold",
        color="#5b2417",
        pad=16,
    )
    ax.set_xlabel("Number of Images", fontsize=13, color="#5b2417")
    ax.set_ylabel("Runtime (seconds)", fontsize=13, color="#5b2417")
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=11, colors="#6d4c41")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c4a484")
    ax.spines["bottom"].set_color("#c4a484")
    ax.grid(True, axis="y", color="#e6d7c3", linewidth=1.0, alpha=0.8)
    ax.grid(False, axis="x")

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#fff7eb",
        edgecolor="#e2c8a6",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if not rows:
        raise SystemExit("No valid run_eval logs found in Eval_loss.")
    summary = summarize(rows)
    save_summary_csv(summary)
    plot(summary)
    print(f"[OK] Saved PNG: {PNG_PATH}")
    print(f"[OK] Saved CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
