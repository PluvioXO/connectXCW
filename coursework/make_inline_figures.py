"""Render the two Q-learning figures at the report's half-column width."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
})


def panels():
    fig, axes = plt.subplots(2, 1, figsize=(2.65, 2.85))
    fig.subplots_adjust(left=.205, right=.977, bottom=.125, top=.823, hspace=.79)
    for ax in axes:
        ax.grid(axis="y", alpha=.20, linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, pad=2)
    return fig, axes


def seed_lines(ax, x, rows, color, label):
    values = 100 * np.array([row["by_seed"] for row in rows])
    ax.plot(x, values, color=color, alpha=.29, linewidth=.75)
    ax.plot(x, [100 * row["mean"] for row in rows], color=color,
            label=label, linewidth=1.5, marker="o", markersize=3)


def save(fig, ax, name):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, .99),
               ncol=2, frameon=False, fontsize=8, handlelength=1.45,
               handletextpad=.45, columnspacing=1.0)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=240)
    plt.close(fig)


def main():
    initial = json.loads((ROOT / "evidence/dqn-study-summary.json").read_text())
    followup = json.loads((ROOT / "evidence/tactical-followup-summary.json").read_text())

    fig, axes = panels()
    for ax, opponent, title in zip(axes, ("random", "minimax"),
                                   ("Uniform random", "Depth-two minimax")):
        for method, color, label in (("dqn", "#087c8b", "DQN"),
                                     ("ddqn", "#7151a1", "Double DQN")):
            seed_lines(ax, range(6), initial["curve"][opponent][method], color, label)
        ax.set(title=title, ylabel="Score (%)", xticks=(0, 1, 2, 3, 4, 5))
    axes[0].set(ylim=(0, 103), yticks=(0, 50, 100))
    axes[1].set(ylim=(0, 20), yticks=(0, 5, 10, 15, 20), xlabel="Self-play games (thousands)")
    save(fig, axes[0], "dqn-study-inline")

    fig, (curve, motifs) = panels()
    groups = ("win-now", "block-now", "multi-step")
    x = np.arange(len(groups))
    for variant, color, label, offset in (("plain", "#326c9a", "Plain DDQN", -.18),
                                          ("tactical", "#bd5709", "Tactical DDQN", .18)):
        seed_lines(curve, (0, 20, 40, 60, 80), followup["learning_curve"][variant], color, label)
        rows = followup["oracle"][variant]["80000"]["greedy"]
        for i, group in enumerate(groups):
            score = rows[group]["optimal_action"]
            motifs.bar(i + offset, 100 * score["mean"], width=.32, color=color, alpha=.85)
            seeds = 100 * np.array(score["by_seed"])
            motifs.scatter(i + offset + np.linspace(-.09, .09, len(seeds)), seeds,
                           color="#182737", s=4, linewidths=0, zorder=3)
    curve.set(title="Depth-two minimax", ylabel="Score (%)", xlabel="Training moves (thousands)",
              xticks=(0, 20, 40, 60, 80), ylim=(0, 16), yticks=(0, 5, 10, 15))
    motifs.set(title="Solved endgames at 80k moves", ylabel="Optimal (%)",
               xticks=x, xticklabels=("Win now", "Defend", "Multi-step"),
               xlim=(-.55, 2.55), ylim=(0, 103), yticks=(0, 50, 100))
    save(fig, curve, "tactical-followup-inline")


if __name__ == "__main__":
    main()
