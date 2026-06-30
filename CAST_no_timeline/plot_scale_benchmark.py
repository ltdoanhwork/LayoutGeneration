#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


TIME_RE = re.compile(r"Total time:\s*([0-9.]+)s")
STEP_RE = re.compile(r"Step 3 \(warp\):\s*([0-9.]+)s")
SCALE_RE = re.compile(r"scale_(\d+)")


def parse_log(log_path: Path) -> tuple[float | None, float | None]:
    total_time = None
    warp_time = None
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in text:
        total_match = TIME_RE.search(line)
        if total_match:
            total_time = float(total_match.group(1))
        step_match = STEP_RE.search(line)
        if step_match:
            warp_time = float(step_match.group(1))
    return total_time, warp_time


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        scale_match = SCALE_RE.search(run_dir.name)
        if not scale_match:
            continue
        scale = int(scale_match.group(1))
        log_path = run_dir / "run_eval.log"
        if not log_path.is_file():
            continue
        total_time, warp_time = parse_log(log_path)
        if total_time is None:
            continue
        rows.append(
            {
                "scale": scale,
                "total_time_sec": total_time,
                "warp_time_sec": warp_time if warp_time is not None else "",
            }
        )
    rows.sort(key=lambda x: x["scale"])
    return rows


def save_csv(rows: list[dict], root: Path) -> Path:
    csv_path = root / "scale_runtime_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scale", "total_time_sec", "warp_time_sec"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot(rows: list[dict], root: Path) -> Path:
    xs = [row["scale"] for row in rows]
    ys = [row["total_time_sec"] for row in rows]
    warp_vals = [
        float(row["warp_time_sec"]) if row["warp_time_sec"] not in ("", None) else None
        for row in rows
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=180)

    fig.patch.set_facecolor("#f7f2ea")
    ax.set_facecolor("#fffaf3")

    y_min = min(ys)
    y_max = max(ys)
    pad = max(2.0, (y_max - y_min) * 0.35)
    baseline = y_min - pad

    ax.fill_between(xs, ys, color="#f0a34a", alpha=0.22, zorder=1)
    ax.plot(
        xs,
        ys,
        color="#b8411f",
        linewidth=3.0,
        marker="o",
        markersize=8,
        markerfacecolor="#fff2cf",
        markeredgecolor="#b8411f",
        markeredgewidth=2.0,
        zorder=3,
        label="Total runtime",
    )

    if all(v is not None for v in warp_vals):
        ax.plot(
            xs,
            warp_vals,
            color="#2f6db3",
            linewidth=2.4,
            marker="s",
            markersize=6.5,
            markerfacecolor="#dcecff",
            markeredgecolor="#2f6db3",
            markeredgewidth=1.6,
            linestyle="--",
            zorder=4,
            label="Warp/render time",
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

    if all(v is not None for v in warp_vals):
        for x, y in zip(xs, warp_vals):
            ax.annotate(
                f"{y:.1f}s",
                (x, y),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=9,
                color="#1f4c7f",
                fontweight="bold",
            )

    ax.set_title(
        "Runtime under Different Scaling Factors",
        fontsize=19,
        fontweight="bold",
        color="#5a2416",
        pad=14,
    )
    ax.set_xlabel("Scaling factor", fontsize=13, color="#5a2416")
    ax.set_ylabel("Runtime (seconds)", fontsize=13, color="#5a2416")
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=11, colors="#6c4b3f")
    ax.set_ylim(baseline, y_max + pad * 0.55)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c8a788")
    ax.spines["bottom"].set_color("#c8a788")
    ax.grid(True, axis="y", color="#e7d7c2", linewidth=1.0, alpha=0.8)
    ax.grid(False, axis="x")

    png_path = root / "scale_runtime_chart.png"
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#fff7eb",
        edgecolor="#e2c8a6",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/layout/scale_benchmark_animals15_dog"
    )
    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"No benchmark logs found in {root}")
    csv_path = save_csv(rows, root)
    png_path = plot(rows, root)
    print(f"[OK] Saved CSV: {csv_path}")
    print(f"[OK] Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
