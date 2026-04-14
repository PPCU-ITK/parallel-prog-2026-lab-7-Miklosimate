#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "scale": int(r["scale"]),
                    "Nx": int(r["Nx"]),
                    "Ny": int(r["Ny"]),
                    "cpu_s": float(r["cpu_s"]),
                    "gpu_s": float(r["gpu_s"]),
                    "speedup": float(r["speedup"]),
                }
            )
    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 plot.py <results.csv> <output.png>")
        sys.exit(1)

    in_csv = sys.argv[1]
    out_png = sys.argv[2]

    rows = read_rows(in_csv)
    labels = [f"{r['scale']}x" for r in rows]
    cpu = [r["cpu_s"] for r in rows]
    gpu = [r["gpu_s"] for r in rows]
    spd = [r["speedup"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax0 = axes[0]
    x = list(range(len(labels)))
    w = 0.35
    ax0.bar([i - w / 2 for i in x], cpu, width=w, label="CPU")
    ax0.bar([i + w / 2 for i in x], gpu, width=w, label="GPU")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_xlabel("Grid scale (Nx, Ny)")
    ax0.set_ylabel("Runtime [s]")
    ax0.set_title("CPU vs GPU runtime")
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(labels, spd, marker="o")
    ax1.set_xlabel("Grid scale (Nx, Ny)")
    ax1.set_ylabel("Speedup (CPU/GPU)")
    ax1.set_title("GPU speedup")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


if __name__ == "__main__":
    main()
