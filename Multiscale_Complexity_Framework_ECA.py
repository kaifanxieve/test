# =========================================================
# MSC-CA : Full Verification Script (FINAL, SAFE VERSION)
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
# 1. Elementary Cellular Automaton
# -----------------------------
def rule_to_table(rule):
    bits = np.binary_repr(rule, width=8)
    return np.array([int(b) for b in bits])[::-1]

def evolve_ca(state, rule_table, rng, p_noise=0.0):
    N = len(state)
    new_state = np.zeros_like(state)
    for i in range(N):
        l = state[(i - 1) % N]
        c = state[i]
        r = state[(i + 1) % N]
        idx = (l << 2) | (c << 1) | r
        new_state[i] = rule_table[idx]
    if p_noise > 0:
        noise = rng.random(N) < p_noise
        new_state = np.logical_xor(new_state, noise).astype(int)
    return new_state

def run_ca(rule, N, T, rng, init="random", p_noise=0.0):
    table = rule_to_table(rule)
    if init == "random":
        state = rng.integers(0, 2, size=N)
    else:
        state = np.zeros(N, dtype=int)
        state[N // 2] = 1

    hist = np.zeros((T, N), dtype=int)
    for t in range(T):
        hist[t] = state
        state = evolve_ca(state, table, rng, p_noise)
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
# 5. Parameters (FINAL, SAFE)
# -----------------------------
N = 512
T = 700
TRANSIENT = 150

RULES = [30, 54, 90, 110]

# ✅ ONLY σ THAT DIVIDE 512
SIGMAS = [1, 2, 4, 8, 16, 32, 64]

REPEATS = 20
P_LIST = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
SIGMA_STAR = 8

# -----------------------------
# 6. Core metric computation
# -----------------------------
def compute_metrics(rule, sigma, p_noise, trial_id):
    rng = rng_for_trial(trial_id)
    hist = run_ca(rule, N, T, rng, p_noise=p_noise)
    hist = hist[TRANSIENT:]

    delta = delta_hamming(hist)

    cg = coarse_grain_history(hist, sigma)
    symbols = cg.reshape(-1)

    H = shannon_entropy(symbols)
    LZ = lz_normalised(symbols.astype(int))

    return {
        "H": H,
        "LZ": LZ,
        "delta_mean": np.mean(delta),
        "delta_std": np.std(delta)
    }

# -----------------------------
# EXP A: Multi-scale complexity
# -----------------------------
C_LZ_mean = {r: [] for r in RULES}
C_LZ_std = {r: [] for r in RULES}

for r in RULES:
    for sigma in SIGMAS:
        vals = []
        for rep in range(REPEATS):
            out = compute_metrics(r, sigma, 0.0, r*10000 + sigma*100 + rep)
            vals.append(out["LZ"])
        C_LZ_mean[r].append(np.mean(vals))
        C_LZ_std[r].append(np.std(vals))

plt.figure(figsize=(9,6))
for r in RULES:
    plt.errorbar(SIGMAS, C_LZ_mean[r], yerr=C_LZ_std[r],
                 marker='o', capsize=3, label=f"Rule {r}")
plt.xlabel("σ (coarse-graining scale)")
plt.ylabel("C(S,σ) (normalised LZ)")
plt.title("Multi-Scale Complexity of ECA (SAFE σ)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# EXP B: Ranking flip
# -----------------------------
print("\nRanking by σ:")
for i, sigma in enumerate(SIGMAS):
    ranking = sorted(
        [(r, C_LZ_mean[r][i]) for r in RULES],
        key=lambda x: x[1], reverse=True
    )
    print(f"σ={sigma}: {[r for r,_ in ranking]}")

# -----------------------------
# EXP C: Noise-induced complexity
# -----------------------------
means, stds = [], []
for p in P_LIST:
    vals = []
    for rep in range(REPEATS):
        out = compute_metrics(110, SIGMA_STAR, p, int(p*1e6)+rep)
        vals.append(out["LZ"])
    means.append(np.mean(vals))
    stds.append(np.std(vals))

plt.figure(figsize=(8,5))
plt.errorbar(P_LIST, means, yerr=stds, marker='o', capsize=3)
plt.xlabel("p (noise probability)")
plt.ylabel(f"C(S,σ={SIGMA_STAR})")
plt.title("Noise-Induced Complexity (Rule 110)")
plt.grid(True)
plt.show()

# -----------------------------
# EXP D: Activity vs Complexity
# -----------------------------
delta_vals, lz_vals = [], []
for r in RULES:
    for rep in range(REPEATS):
        out = compute_metrics(r, SIGMA_STAR, 0.0, r*5000+rep)
        delta_vals.append(out["delta_mean"])
        lz_vals.append(out["LZ"])

def pearson(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.sum(x*y) / np.sqrt(np.sum(x*x)*np.sum(y*y))

print("\nPearson corr(Δ̄, C_LZ):", pearson(np.array(delta_vals), np.array(lz_vals)))

plt.figure(figsize=(6,5))
plt.scatter(delta_vals, lz_vals)
plt.xlabel("Δ̄ (activity)")
plt.ylabel("C_LZ")
plt.title("Activity vs Complexity")
plt.grid(True)
plt.show()

print("\nDONE: All experiments completed successfully.")
