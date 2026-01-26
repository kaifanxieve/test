# =========================================================
# MSC-Framework : Reaction–Diffusion Verification (PATH 2)
# System: 1D Reaction–Diffusion (Gray–Scott, discrete)
# Platform: Google Colab / Python 3
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

# -----------------------------
# 0. Reproducibility
# -----------------------------
BASE_SEED = 20260126

def rng_for_trial(trial_id):
    return np.random.default_rng(BASE_SEED + trial_id)

# -----------------------------
# 1. 1D Reaction–Diffusion System
# -----------------------------
def laplacian_1d(x):
    return np.roll(x, 1) + np.roll(x, -1) - 2*x

def run_rd(N, T, rng, Du=0.16, Dv=0.08, F=0.035, k=0.065, p_noise=0.0):
    u = np.ones(N)
    v = np.zeros(N)

    # small random perturbation
    v[N//2-5:N//2+5] = 1.0
    u -= v

    hist = np.zeros((T, N))
    for t in range(T):
        hist[t] = (v > 0.5).astype(int)

        Lu = laplacian_1d(u)
        Lv = laplacian_1d(v)

        uvv = u * v * v
        u += Du * Lu - uvv + F * (1 - u)
        v += Dv * Lv + uvv - (F + k) * v

        if p_noise > 0:
            v += p_noise * rng.normal(0, 0.1, size=N)

        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

    return hist

# -----------------------------
# 2. Δt (activity)
# -----------------------------
def delta_hamming(history):
    return np.mean(history[:-1] != history[1:], axis=1)

# -----------------------------
# 3. Coarse-graining (SAFE)
# -----------------------------
def coarse_grain_row(row, sigma):
    N = len(row)
    if N % sigma != 0:
        raise ValueError("sigma must divide N exactly")
    blocks = N // sigma
    return np.array([
        1 if np.mean(row[i*sigma:(i+1)*sigma]) >= 0.5 else 0
        for i in range(blocks)
    ])

def coarse_grain_history(history, sigma):
    return np.array([coarse_grain_row(row, sigma) for row in history])

# -----------------------------
# 4. Complexity Measures
# -----------------------------
def shannon_entropy(symbols):
    counts = Counter(symbols.tolist())
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H

def lempel_ziv_complexity(seq):
    s = seq.tolist()
    n = len(s)
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if l + k > n:
            c += 1
            break
        if s[i+k-1] == s[l+k-1]:
            k += 1
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    i, k = 0, 1
            else:
                l += 1
        if l >= n:
            break
    return c

def lz_normalised(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    c = lempel_ziv_complexity(seq)
    return c / (n / math.log2(n))

# -----------------------------
# 5. Parameters
# -----------------------------
N = 512
T = 700
TRANSIENT = 150

SIGMAS = [1, 2, 4, 8, 16, 32, 64]
REPEATS = 20
SIGMA_STAR = 8
P_LIST = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]

# -----------------------------
# 6. Core metric computation
# -----------------------------
def compute_metrics(sigma, p_noise, trial_id):
    rng = rng_for_trial(trial_id)
    hist = run_rd(N, T, rng, p_noise=p_noise)
    hist = hist[TRANSIENT:]

    delta = delta_hamming(hist)

    cg = coarse_grain_history(hist, sigma)
    symbols = cg.reshape(-1)

    return {
        "H": shannon_entropy(symbols),
        "LZ": lz_normalised(symbols.astype(int)),
        "delta_mean": np.mean(delta),
        "delta_std": np.std(delta)
    }

# -----------------------------
# EXP A: Multi-scale complexity
# -----------------------------
C_LZ_mean, C_LZ_std = [], []

for sigma in SIGMAS:
    vals = []
    for rep in range(REPEATS):
        out = compute_metrics(sigma, 0.0, sigma*100 + rep)
        vals.append(out["LZ"])
    C_LZ_mean.append(np.mean(vals))
    C_LZ_std.append(np.std(vals))

plt.figure(figsize=(9,6))
plt.errorbar(SIGMAS, C_LZ_mean, yerr=C_LZ_std,
             marker='o', capsize=3)
plt.xlabel("σ (coarse-graining scale)")
plt.ylabel("C(S,σ) (normalised LZ)")
plt.title("Multiscale Complexity — Reaction–Diffusion System")
plt.grid(True)
plt.show()

# -----------------------------
# EXP B: Noise-induced complexity
# -----------------------------
means, stds = [], []
for p in P_LIST:
    vals = []
    for rep in range(REPEATS):
        out = compute_metrics(SIGMA_STAR, p, int(p*1e6)+rep)
        vals.append(out["LZ"])
    means.append(np.mean(vals))
    stds.append(np.std(vals))

plt.figure(figsize=(8,5))
plt.errorbar(P_LIST, means, yerr=stds, marker='o', capsize=3)
plt.xlabel("p (noise strength)")
plt.ylabel(f"C(S,σ={SIGMA_STAR})")
plt.title("Noise–Structure Interaction (Reaction–Diffusion)")
plt.grid(True)
plt.show()

# -----------------------------
# EXP C: Activity vs Complexity
# -----------------------------
delta_vals, lz_vals = [], []
for rep in range(REPEATS):
    out = compute_metrics(SIGMA_STAR, 0.0, rep)
    delta_vals.append(out["delta_mean"])
    lz_vals.append(out["LZ"])

def pearson(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.sum(x*y) / np.sqrt(np.sum(x*x)*np.sum(y*y))

print("Pearson corr(Δ̄, C_LZ):", pearson(np.array(delta_vals), np.array(lz_vals)))

plt.figure(figsize=(6,5))
plt.scatter(delta_vals, lz_vals)
plt.xlabel("Δ̄ (activity)")
plt.ylabel("C_LZ")
plt.title("Activity vs Complexity (Reaction–Diffusion)")
plt.grid(True)
plt.show()

print("\nDONE: Reaction–Diffusion multiscale experiment completed.")
