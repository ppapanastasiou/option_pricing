"""Plot Black-Scholes price and greeks against spot for a call and a put."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from black_scholes import black_scholes

CALL_COLOR = "#2a78d6"
PUT_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS = [
    ("price", "Price"),
    ("delta", "Delta"),
    ("gamma", "Gamma"),
    ("vega", "Vega (per 1 vol pt)"),
    ("theta", "Theta (per day)"),
    ("rho", "Rho (per 1% rate)"),
]


def compute_curves(spots, K, T, r, sigma, q):
    curves = {}
    for kind in ("call", "put"):
        rows = [black_scholes(S, K, T, r, sigma, kind, q) for S in spots]
        curves[kind] = {key: np.array([row[key] for row in rows]) for key, _ in PANELS}
    return curves


def plot_greeks(K=100, T=1.0, r=0.05, sigma=0.20, q=0.0, out="greeks.png"):
    spots = np.linspace(0.5 * K, 1.5 * K, 301)
    curves = compute_curves(spots, K, T, r, sigma, q)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
    for ax, (key, label) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.plot(spots, curves["call"][key], color=CALL_COLOR, lw=2, label="Call")
        # Gamma and vega are identical for calls and puts; dashes keep both visible.
        ax.plot(spots, curves["put"][key], color=PUT_COLOR, lw=2, ls="--", label="Put")
        ax.axvline(K, color=INK_SECONDARY, lw=1, ls=":")
        ax.axhline(0, color=GRID, lw=1)
        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("Spot", color=INK_SECONDARY)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle(
        f"Black-Scholes greeks vs spot   K={K}, T={T}y, r={r:.1%}, vol={sigma:.0%}, q={q:.1%}",
        color=INK, fontsize=14, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K", type=float, default=100, help="strike")
    p.add_argument("--T", type=float, default=1.0, help="years to expiry")
    p.add_argument("--r", type=float, default=0.05, help="risk-free rate")
    p.add_argument("--sigma", type=float, default=0.20, help="volatility")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield")
    p.add_argument("--out", default="greeks.png", help="output image file")
    a = p.parse_args()
    plot_greeks(a.K, a.T, a.r, a.sigma, a.q, a.out)
