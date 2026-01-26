# =========================================================
# MSC-Framework : Agent-Based Verification (PATH 2)
# System: 1D Agent-Based Model (Local Majority + Conflict Noise)
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
# 1. Agent-Based System
# -----------------------------
def run_agent_system(N, T, rng, p_noise=0.0):
    state = rng.integers(0, 2, size=N)
    hist = np.zeros((T, N), dtype=int)

    for t in range(T):
        hist[t] = state
        new_state = state.copy()
        for i in range(N):
            nbrs = state[(i-1) % N], state[i], state[(i+1) % N]
            if sum(nbrs) >= 2:
                new_state[i] = 1
            else:
                new_state[i] = 0
        if p_noise > 0:
            noise = rng.random(N) < p_noise
            new_state = np.logical_xor(new_state, noise).astype(int)
        state = new_state
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
P_LIST = [0.0, 0.001, 0.002, 0.005, 0.01, 0.05]

# -----------------------------
# 6. Core metric computation
# -----------------------------
def compute_metrics(sigma, p_noise, trial_id):
    rng = rng_for_trial(trial_id)
    hist = run_agent_system(N, T, rng, p_noise=p_noise)
    hist = hist[TRANSIENT:]

    delta = delta_hamming(hist)
    cg = coarse_grain_history(hist, sigma)
    symbols = cg.reshape(-1)

    return {
        "LZ": lz_normalised(symbols.astype(int)),
        "delta_mean": np.mean(delta)
    }

# -----------------------------
# EXP A: Multi-scale complexity
# -----------------------------
C_mean, C_std = [], []
for sigma in SIGMAS:
    vals = []
    for rep in range(REPEATS):
        out = compute_metrics(sigma, 0.0, sigma*100 + rep)
        vals.append(out["LZ"])
    C_mean.append(np.mean(vals))
    C_std.append(np.std(vals))

plt.figure(figsize=(8,5))
plt.errorbar(SIGMAS, C_mean, yerr=C_std, marker='o', capsize=3)
plt.xlabel("σ (coarse-graining scale)")
plt.ylabel("C(S,σ) (normalised LZ)")
plt.title("Multiscale Complexity — Agent-Based System")
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

plt.figure(figsize=(7,4))
plt.errorbar(P_LIST, means, yerr=stds, marker='o', capsize=3)
plt.xlabel("p (noise probability)")
plt.ylabel(f"C(S,σ={SIGMA_STAR})")
plt.title("Noise–Structure Interaction (Agent-Based)")
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
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.sum(x*y) / np.sqrt(np.sum(x*x)*np.sum(y*y))

print("Pearson corr(Δ̄, C_LZ):", pearson(np.array(delta_vals), np.array(lz_vals)))

plt.figure(figsize=(6,5))
plt.scatter(delta_vals, lz_vals)
plt.xlabel("Δ̄ (activity)")
plt.ylabel("C_LZ")
plt.title("Activity vs Complexity (Agent-Based)")
plt.grid(True)
plt.show()

print("\nDONE: Agent-based multiscale experiment completed.")
