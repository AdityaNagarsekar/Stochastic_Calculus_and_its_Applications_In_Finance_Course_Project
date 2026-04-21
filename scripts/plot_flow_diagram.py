"""
Clean project flow diagram — strict grid, L-shaped arrows, no overlaps.
Saves results/project_flow_diagram.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 15, 10
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── palette ───────────────────────────────────────────────────────────────────
BLUE   = ("#dce8f5", "#2471a3")   # data
GREEN  = ("#d5ecd4", "#1e8449")   # P3 / LSTM
AMBER  = ("#fef3cd", "#b7770d")   # P1 / RL
PURPLE = ("#ead5ec", "#7d3c98")   # integration
DARK   = "#2c3e50"

# ── primitives ────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, title, sub=None, fc="white", ec="black",
        bold=False, fs=8.5, zorder=3):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.07",
                       facecolor=fc, edgecolor=ec, lw=1.7, zorder=zorder)
    ax.add_patch(p)
    dy = 0.12 if sub else 0
    ax.text(cx, cy+dy, title, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=zorder+1)
    if sub:
        ax.text(cx, cy-0.19, sub, ha="center", va="center",
                fontsize=fs-1.5, color="#555", style="italic", zorder=zorder+1)

def horiz(x0, x1, y, color=DARK, lw=1.5):
    """Straight horizontal arrow."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10), zorder=2)

def vert(x, y0, y1, color=DARK, lw=1.5):
    """Straight vertical arrow."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10), zorder=2)

def elbow(x0, y0, x1, y1, color=DARK, lw=1.5, via="v"):
    """
    L-shaped connector with arrowhead at destination.
    via='v' : go vertical first (x0,y0)→(x0,y1)→(x1,y1)
    via='h' : go horizontal first (x0,y0)→(x1,y0)→(x1,y1)
    """
    if via == "v":
        ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=2,
                solid_capstyle="round")
        ax.annotate("", xy=(x1, y1),
                    xytext=(x1 - 0.001*(x1-x0 or 1e-3), y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=10), zorder=2)
    else:  # h
        ax.plot([x0, x1, x1], [y0, y0, y1], color=color, lw=lw, zorder=2,
                solid_capstyle="round")
        ax.annotate("", xy=(x1, y1),
                    xytext=(x1, y1 + 0.001*(y1-y0 or 1e-3)),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=10), zorder=2)

def dot(cx, cy, color=DARK):
    ax.plot(cx, cy, "o", color=color, ms=8, zorder=5)

def label(x, y, txt, color=DARK, fs=7.0):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)

def section(x, y, w, h, color, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor=color,
                       alpha=0.13, lw=0, zorder=1)
    ax.add_patch(p)
    ax.text(x+0.18, y+h-0.18, title,
            fontsize=7.5, fontweight="bold", color=color,
            va="top", zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS  (drawn first so boxes sit on top)
# ══════════════════════════════════════════════════════════════════════════════
section(0.2,  8.1, 14.6, 1.65, BLUE[1],   "DATA")
section(0.2,  4.6, 6.4,  3.25, GREEN[1],  "PROJECT 3 — LSTM VOLATILITY")
section(8.4,  4.6, 6.4,  3.25, AMBER[1],  "PROJECT 1 — RL HEDGING")
section(0.2,  0.5, 14.6, 3.85, PURPLE[1], "INTEGRATION")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — DATA  (y = 8.90)
# ══════════════════════════════════════════════════════════════════════════════
#  ┌─────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐
#  │ Nifty 50 Prices │→ │  Feature Engineering      │→ │ Train / Val / Test │
#  └─────────────────┘  └──────────────────────────┘  └────────────────────┘
box(2.2,  8.90, 3.0, 0.72, "Nifty 50 Prices",
    "Yahoo Finance 2010–2024", *BLUE, bold=True)
box(6.5,  8.90, 4.4, 0.72, "Feature Engineering",
    "log_ret · rv5 · rv21 · rv_ratio · mom5   →   X:(3597,60,6)", *BLUE)
box(12.0, 8.90, 3.6, 0.72, "Train / Val / Test Splits",
    "70% / 15% / 15%", *BLUE)

horiz(3.70, 4.30, 8.90, color=BLUE[1])
horiz(8.70, 10.20, 8.90, color=BLUE[1])

# ══════════════════════════════════════════════════════════════════════════════
# FORK from Splits → P3 (left) and P1 (right)
# ══════════════════════════════════════════════════════════════════════════════
# Both forks leave from the bottom of the Splits box (12.0, 8.54)
# P3 → elbow: down to y=7.92, left to x=2.3
# P1 → elbow: down to y=7.92, right to x=10.5
elbow(12.0, 8.54, 2.3, 7.92, color=BLUE[1], lw=1.5, via="v")
elbow(12.0, 8.54, 10.5, 7.92, color=BLUE[1], lw=1.5, via="v")
label(7.0, 7.97, "to P3", color=BLUE[1])
label(12.5, 7.97, "to P1", color=BLUE[1])

# ══════════════════════════════════════════════════════════════════════════════
# P3 LEFT TRACK  (x = 0.2–6.6, y = 4.6–7.85)
# ══════════════════════════════════════════════════════════════════════════════
#  ┌──────────────────┐
#  │  Stacked LSTM    │───╮
#  └──────────────────┘   ●──→ ┌──────────────────┐
#  ┌──────────────────┐   │    │   P3 Results     │
#  │   GARCH(1,1)     │───╯    └──────────────────┘
#  └──────────────────┘
#  ┌──────────────────┐
#  │ BSM + CRR Binom  │────────── (feeds pricing inside P3 Results)
#  └──────────────────┘

box(2.3, 7.35, 3.2, 0.72, "Stacked LSTM",
    "LSTM(64)→LSTM(32)→Softplus", *GREEN, bold=True)
box(2.3, 6.25, 3.2, 0.72, "GARCH(1,1) Benchmark",
    "rolling out-of-sample", *GREEN)
box(2.3, 5.15, 3.2, 0.72, "BSM  +  CRR Binomial",
    "option pricing & N-convergence", *GREEN)

# merge dot
dot(4.60, 6.80, GREEN[1])
horiz(3.90, 4.60, 7.35, color=GREEN[1])
horiz(3.90, 4.60, 6.25, color=GREEN[1])
horiz(3.90, 4.60, 5.15, color=GREEN[1])

# P3 Results box
box(6.0, 6.10, 2.8, 1.80,
    "P3 Results",
    "LSTM RMSE=0.019\nGARCH RMSE=0.025\nLSTM beats GARCH ✓\nPricing MAE table\nCRR→BSM convergence",
    *AMBER, fs=7.8)
horiz(4.60, 4.60, 6.80, color=GREEN[1])  # merge dot is at (4.60, 6.80)
# now from merge dot to P3 results box left edge (4.60, 6.80) → (4.60, 6.10)→ (5.6, 6.10)?
# Better: direct horiz from dot to left edge of P3 Results
# dot at (4.60, 6.80), P3 Results left edge at (6.0-1.4=4.6, 6.10)
# Need to draw line from (4.60,6.80) to (4.60, 6.10) then to (4.60 → 5.60, 6.10)
# Actually let's just draw from dot diagonally... or elbow:
# elbow from (4.60, 6.80) down to (4.60, 6.10), then right to (5.60, 6.10)
# That's within the already-drawn horiz line at y=6.80... let me redo:

# ══════════════════════════════════════════════════════════════════════════════
# P1 RIGHT TRACK  (x = 8.4–14.8, y = 4.6–7.85)
# ══════════════════════════════════════════════════════════════════════════════
box(10.5, 7.35, 3.0, 0.72, "HedgingEnv  +  PPO",
    "GBM σ=0.20 · PPO 500k steps", *AMBER, bold=True)
box(10.5, 6.25, 3.0, 0.72, "BSM Baselines",
    "daily & weekly rebalancing", *AMBER)

dot(13.0, 6.80, AMBER[1])
horiz(12.0, 13.0, 7.35, color=AMBER[1])
horiz(12.0, 13.0, 6.25, color=AMBER[1])

box(14.1, 6.10, 2.8, 1.80,
    "P1 Results",
    "corr(RL, BSM)=0.96\nRL learns BSM delta ✓\nBSM MAE=0.28\nRL MAE=0.75\nTC comparison",
    *AMBER, fs=7.8)
horiz(13.0, 12.20, 6.80, color=AMBER[1])

# ══════════════════════════════════════════════════════════════════════════════
# Fix: redo P3 merge properly
# ══════════════════════════════════════════════════════════════════════════════
# The three horiz arrows from model right edges to dot should go:
# model right edges:  (2.3+1.6=3.9, 7.35), (3.9, 6.25), (3.9, 5.15)
# merge dot: (4.60, 6.80) — but this doesn't line up simply
# Let me instead route them as elbows to the dot:
# LSTM right (3.9, 7.35) → elbow right then down → (4.60, 7.35)→(4.60, 6.80)
# GARCH right (3.9, 6.25) → straight right → (4.60, 6.25)→ up to (4.60, 6.80)
# BSM right (3.9, 5.15) → elbow right then up → (4.60, 5.15)→(4.60, 6.80)
# Then dot → P3 Results: (4.60, 6.80) → right to (5.60, 6.80)→ down to (5.60, 6.10+0.90=7.0)
# Hmm, this is still complicated. Let me just redraw all arrows cleanly:

# ══════════════════════════════════════════════════════════════════════════════
# REDO the whole thing with a cleaner, simpler approach.
# ══════════════════════════════════════════════════════════════════════════════

plt.close(fig)  # discard what we drew above

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ── re-define primitives for clean ax ────────────────────────────────────────
def box(cx, cy, w, h, title, sub=None, fc="white", ec="black",
        bold=False, fs=8.5):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.07",
                       facecolor=fc, edgecolor=ec, lw=1.8, zorder=3)
    ax.add_patch(p)
    dy = 0.13 if sub else 0
    ax.text(cx, cy+dy, title, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)
    if sub:
        ax.text(cx, cy-0.19, sub, ha="center", va="center",
                fontsize=fs-1.5, color="#555", style="italic", zorder=4)
    # return edges: left, right, top, bottom, center
    return dict(L=(cx-w/2, cy), R=(cx+w/2, cy),
                T=(cx, cy+h/2), B=(cx, cy-h/2), C=(cx, cy))

def arr(src, dst, color=DARK, lw=1.5, label_txt=""):
    """Straight arrow from point to point."""
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=11), zorder=2)
    if label_txt:
        mx, my = (src[0]+dst[0])/2, (src[1]+dst[1])/2
        ax.text(mx, my+0.14, label_txt, ha="center", va="center",
                fontsize=6.5, color=color,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)

def line(pts, color=DARK, lw=1.5):
    """Polyline (no arrowhead)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")

def line_arr(pts, color=DARK, lw=1.5, label_txt=""):
    """Polyline with arrowhead at the last segment."""
    line(pts[:-1] + [pts[-1]], color, lw)  # draw whole path
    # arrowhead at last segment
    arr(pts[-2], pts[-1], color=color, lw=lw)
    if label_txt:
        # label near midpoint of path
        n = len(pts)
        i = n // 2
        mx, my = (pts[i][0]+pts[i-1][0])/2, (pts[i][1]+pts[i-1][1])/2
        ax.text(mx, my+0.15, label_txt, ha="center", va="center",
                fontsize=6.5, color=color,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)

def dot(cx, cy, color=DARK):
    ax.plot(cx, cy, "o", color=color, ms=7, zorder=5)

def section(x, y, w, h, color, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor=color,
                       alpha=0.13, lw=0, zorder=1)
    ax.add_patch(p)
    ax.text(x+0.2, y+h-0.18, title,
            fontsize=8, fontweight="bold", color=color,
            va="top", zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════════
section(0.2, 8.1,  14.6, 1.65, BLUE[1],   "DATA")
section(0.2, 4.55, 6.55, 3.3,  GREEN[1],  "PROJECT 3 — LSTM VOLATILITY")
section(8.25,4.55, 6.55, 3.3,  AMBER[1],  "PROJECT 1 — RL HEDGING")
section(0.2, 0.4,  14.6, 3.9,  PURPLE[1], "INTEGRATION")

# ══════════════════════════════════════════════════════════════════════════════
# DATA ROW  y = 8.9
# ══════════════════════════════════════════════════════════════════════════════
b_nifty = box(2.2,  8.9, 3.0, 0.72,
              "Nifty 50 Prices", "Yahoo Finance  2010–2024", *BLUE, bold=True)
b_feat  = box(7.5,  8.9, 5.8, 0.72,
              "Feature Engineering  →  Sequences  →  Splits",
              "log_ret · rv5 · rv21 · rv_ratio · mom5   |   X:(3597,60,6)   |   70/15/15%",
              *BLUE)

arr(b_nifty["R"], b_feat["L"], color=BLUE[1])

# fork from bottom of Feature box — left to P3, right to P1
fork_x   = 7.5          # x of fork (centre of feature box)
fork_y   = b_feat["B"][1]   # = 8.54

# To P3 LSTM (will be at cx=2.4, cy=7.5)
line_arr([(fork_x, fork_y), (fork_x, 8.1), (2.4, 8.1), (2.4, 7.87)],
         color=BLUE[1], label_txt="sequences")
# To P1 HedgingEnv (will be at cx=10.0, cy=7.5)
line_arr([(fork_x, fork_y), (fork_x, 8.1), (10.0, 8.1), (10.0, 7.87)],
         color=BLUE[1])

# ══════════════════════════════════════════════════════════════════════════════
# P3 TRACK  (x centre ≈ 2.4–6.5, y ≈ 4.7–7.8)
# ══════════════════════════════════════════════════════════════════════════════
# Three models on left, merge → results on right
BW, BH = 3.0, 0.70   # standard box size for models

b_lstm  = box(2.4, 7.35, BW, BH, "Stacked LSTM",
              "LSTM(64)→LSTM(32)→Softplus", *GREEN, bold=True)
b_garch = box(2.4, 6.30, BW, BH, "GARCH(1,1)",
              "rolling out-of-sample benchmark", *GREEN)
b_crr   = box(2.4, 5.25, BW, BH, "BSM  +  CRR Binomial",
              "option pricing & convergence table", *GREEN)

# merge dot at x=4.6, y=6.30 (same height as GARCH)
M3 = (4.6, 6.30)
dot(*M3, color=GREEN[1])

# LSTM right → M3 (elbow: right then down)
line_arr([(b_lstm["R"][0],  7.35), (4.6, 7.35), M3], color=GREEN[1])
# GARCH right → M3 (straight)
arr(b_garch["R"], M3, color=GREEN[1])
# CRR right → M3 (elbow: right then up)
line_arr([(b_crr["R"][0],   5.25), (4.6, 5.25), M3], color=GREEN[1])

b_p3res = box(6.3, 6.30, 2.9, 2.10,
              "P3 Results",
              "LSTM RMSE=0.019\nGARCH RMSE=0.025\nLSTM beats GARCH ✓\nPricing MAE comparison\nCRR→BSM convergence",
              *AMBER, fs=7.8)
arr(M3, b_p3res["L"], color=GREEN[1])

# ══════════════════════════════════════════════════════════════════════════════
# P1 TRACK  (x centre ≈ 9.5–14.8, y ≈ 4.7–7.8)
# ══════════════════════════════════════════════════════════════════════════════
b_env  = box(10.0, 7.35, 3.0, BH, "HedgingEnv",
             "GBM · σ=0.20 · κ=0.001", *AMBER, bold=True)
b_ppo  = box(10.0, 6.30, 3.0, BH, "PPO Agent",
             "500k steps · MLP[128, 64]", *AMBER)
b_bsm  = box(10.0, 5.25, 3.0, BH, "BSM Baselines",
             "daily & weekly rebalancing", *AMBER)

M1 = (12.3, 6.30)
dot(*M1, color=AMBER[1])

line_arr([(b_env["R"][0],  7.35), (12.3, 7.35), M1], color=AMBER[1])
arr(b_ppo["R"], M1, color=AMBER[1])
line_arr([(b_bsm["R"][0],  5.25), (12.3, 5.25), M1], color=AMBER[1])

b_p1res = box(14.1, 6.30, 2.9, 2.10,
              "P1 Results",
              "corr(RL, BSM)=0.96\nRL learns BSM delta ✓\nBSM MAE=0.28\nRL MAE=0.75\nTC comparison",
              *AMBER, fs=7.8)
arr(M1, b_p1res["L"], color=AMBER[1])

# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION ROW  (y ≈ 1.0–4.3)
# ══════════════════════════════════════════════════════════════════════════════
b_pool = box(2.2,  2.80, 3.0, BH, "LSTM Vol Pool",
             "3046 predictions ∈ [7%, 80%]", *PURPLE, bold=True)
b_lenv = box(5.9,  2.80, 3.2, BH, "LSTMHedgingEnv",
             "σ drawn from LSTM dist. per episode", *PURPLE, bold=True)
b_lrl  = box(9.8,  2.80, 2.8, BH, "LSTM-RL Agent",
             "PPO 500k · vol-aware", *PURPLE)

# merge dot for final comparison: LSTM-RL + baseline RL
Mf = (12.1, 2.80)
dot(*Mf, color=DARK)

arr(b_pool["R"], b_lenv["L"], color=PURPLE[1])
arr(b_lenv["R"], b_lrl["L"],  color=PURPLE[1])
arr(b_lrl["R"],  Mf,           color=PURPLE[1], label_txt="LSTM-RL")

b_final = box(13.7, 2.30, 2.8, 1.65,
              "Final Comparison",
              "BSM daily   MAE=0.28\nRL (σ=0.20) MAE=0.78\nRL (LSTM σ) TC=0.10 ↓\nvol-aware→lower cost",
              fc="#f0f4f8", ec=DARK, fs=8.0)
arr(Mf, b_final["L"], color=DARK)

# ── Cross-section connectors ──────────────────────────────────────────────────
# P3 Results → LSTM Vol Pool  (vol predictions flow down)
line_arr([(b_p3res["B"][0], b_p3res["B"][1]),
          (b_p3res["B"][0], 3.8),
          (b_pool["T"][0],  3.8),
          b_pool["T"]],
         color=PURPLE[1], label_txt="vol predictions")

# P1 HedgingEnv bottom → LSTMHedgingEnv top  (env architecture inherited)
line_arr([(b_env["B"][0],  b_env["B"][1]),
          (b_env["B"][0],  4.3),
          (b_lenv["T"][0], 4.3),
          b_lenv["T"]],
         color=PURPLE[1], label_txt="env structure")

# P1 Results → final merge dot  (baseline RL performance)
line_arr([(b_p1res["B"][0], b_p1res["B"][1]),
          (b_p1res["B"][0], 1.55),
          (Mf[0],           1.55),
          Mf],
         color=AMBER[1], label_txt="baseline RL")

# ══════════════════════════════════════════════════════════════════════════════
# TITLE + LEGEND
# ══════════════════════════════════════════════════════════════════════════════
ax.text(7.5, 9.72,
        "Stochastic Calculus Project — Architecture & Data Flow",
        ha="center", va="center", fontsize=12.5,
        fontweight="bold", color=DARK)

legend_items = [
    mpatches.Patch(fc=BLUE[0],   ec=BLUE[1],   label="Data"),
    mpatches.Patch(fc=GREEN[0],  ec=GREEN[1],  label="Model / Algorithm"),
    mpatches.Patch(fc=AMBER[0],  ec=AMBER[1],  label="Output / Result"),
    mpatches.Patch(fc=PURPLE[0], ec=PURPLE[1], label="Integration"),
]
ax.legend(handles=legend_items, loc="lower right",
          bbox_to_anchor=(0.99, 0.01), fontsize=8.5,
          framealpha=0.95, ncol=4)

plt.tight_layout(pad=0.3)
fig.savefig(RESULTS / "project_flow_diagram.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved results/project_flow_diagram.png")
