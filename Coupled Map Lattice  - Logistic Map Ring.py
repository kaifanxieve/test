# =========================================================
# MSC-Framework : Coupled Map Lattice (Logistic Map Ring)
# 
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
def rng_for_trial(trial_id: int):
    return np.random.default_rng(BASE_SEED + int(trial_id))

# -----------------------------
# 1. Coupled Map Lattice (Logistic Ring)
# -----------------------------
def logistic_map(x, r):
    return r * x * (1.0 - x)

def run_cml_logistic_ring(
    N: int,
    T: int,
    rng,
    r: float = 3.9,
    eps: float = 0.25,
    p_noise: float = 0.0,
    noise_sigma: float = 0.02,
):
    """
    x_{t+1}(i) = (1-eps) f(x_t(i)) + (eps/2) [ f(x_t(i-1)) + f(x_t(i+1)) ]
    Periodic boundary.
    Optional noise: with prob p_noise, add N(0, noise_sigma) to x after update.
    """
    x = rng.random(N)  # in (0,1)
    hist_bin = np.zeros((T, N), dtype=int)

    for t in range(T):
        # binary observation (for MSC pipeline)
        hist_bin[t] = (x >= 0.5).astype(int)

        fx = logistic_map(x, r)
        fx_l = np.roll(fx, 1)
        fx_r = np.roll(fx, -1)

        x_next = (1.0 - eps) * fx + (eps * 0.5) * (fx_l + fx_r)

        if p_noise > 0:
            mask = rng.random(N) < p_noise
            x_next = x_next + mask * rng.normal(0.0, noise_sigma, size=N)

        # keep in [0,1]
        x = np.clip(x_next, 0.0, 1.0)

    return hist_bin

# -----------------------------
# 2. Δt (activity)
# -----------------------------
def delta_hamming(history_bin):
    return np.mean(history_bin[:-1] != history_bin[1:], axis=1)

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
    ], dtype=int)

def coarse_grain_history(history, sigma):
    return np.array([coarse_grain_row(row, sigma) for row in history], dtype=int)

# -----------------------------
# 4. Lempel–Ziv Complexity
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

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.sum(x*y) / np.sqrt(np.sum(x*x) * np.sum(y*y))

# -----------------------------
# 5. Parameters (SAFE)
# -----------------------------
N = 512
T = 700
TRANSIENT = 150

# Only σ that divide 512
SIGMAS = [1, 2, 4, 8, 16, 32, 64]
REPEATS = 20

# CML parameters
EPS = 0.25
R_LIST = [3.5, 3.7, 3.9, 4.0]   # treat like "rules" for ranking/flip tests
R_STAR = 3.9
SIGMA_STAR = 8

P_LIST = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
NOISE_SIGMA = 0.02

# -----------------------------
# 6. Core metric computation
# -----------------------------
def compute_metrics(r, sigma, p_noise, trial_id):
    rng = rng_for_trial(trial_id)
    hist = run_cml_logistic_ring(
        N=N, T=T, rng=rng, r=r, eps=EPS,
        p_noise=p_noise, noise_sigma=NOISE_SIGMA
    )
    hist = hist[TRANSIENT:]

    delta = delta_hamming(hist)

    cg = coarse_grain_history(hist, sigma)
    symbols = cg.reshape(-1)

    return {
        "LZ": lz_normalised(symbols.astype(int)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
    }

# -----------------------------
# EXP A: Multi-scale complexity (across r)
# -----------------------------
C_LZ_mean = {r: [] for r in R_LIST}
C_LZ_std  = {r: [] for r in R_LIST}

for r in R_LIST:
    for sigma in SIGMAS:
        vals = []
        for rep in range(REPEATS):
            trial_id = int(r*100000) + sigma*100 + rep
            out = compute_metrics(r, sigma, 0.0, trial_id)
            vals.append(out["LZ"])
        C_LZ_mean[r].append(np.mean(vals))
        C_LZ_std[r].append(np.std(vals))

plt.figure(figsize=(9,6))
for r in R_LIST:
    plt.errorbar(SIGMAS, C_LZ_mean[r], yerr=C_LZ_std[r],
                 marker='o', capsize=3, label=f"r={r}")
plt.xlabel("σ (coarse-graining scale)")
plt.ylabel("C(S,σ) (normalised LZ)")
plt.title("Multiscale Complexity — CML Logistic Ring")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# EXP B: Ranking flip by σ (across r)
# -----------------------------
print("\nRanking by σ (higher C_LZ first):")
for i, sigma in enumerate(SIGMAS):
    ranking = sorted([(r, C_LZ_mean[r][i]) for r in R_LIST], key=lambda x: x[1], reverse=True)
    print(f"σ={sigma}: {[r for r,_ in ranking]}")

# -----------------------------
# EXP C: Noise-induced complexity (fixed r*, σ*)
# -----------------------------
means, stds = [], []
for p in P_LIST:
    vals = []
    for rep in range(REPEATS):
        trial_id = int(p*1e6) + rep + 999000
        out = compute_metrics(R_STAR, SIGMA_STAR, p, trial_id)
        vals.append(out["LZ"])
    means.append(np.mean(vals))
    stds.append(np.std(vals))

plt.figure(figsize=(8,5))
plt.errorbar(P_LIST, means, yerr=stds, marker='o', capsize=3)
plt.xlabel("p (noise probability)")
plt.ylabel(f"C(S,σ={SIGMA_STAR})")
plt.title(f"Noise–Structure Interaction — CML (r={R_STAR}, eps={EPS})")
plt.grid(True)
plt.show()

# -----------------------------
# EXP D: Activity vs Complexity (across r, repeats)
# -----------------------------
delta_vals, lz_vals = [], []
for r in R_LIST:
    for rep in range(REPEATS):
        trial_id = int(r*200000) + rep + 12345
        out = compute_metrics(r, SIGMA_STAR, 0.0, trial_id)
        delta_vals.append(out["delta_mean"])
        lz_vals.append(out["LZ"])

print("\nPearson corr(Δ̄, C_LZ):", pearson(np.array(delta_vals), np.array(lz_vals)))

plt.figure(figsize=(6,5))
plt.scatter(delta_vals, lz_vals)
plt.xlabel("Δ̄ (activity)")
plt.ylabel("C_LZ")
plt.title(f"Activity vs Complexity — CML (σ={SIGMA_STAR})")
plt.grid(True)
plt.show()

print("\nDONE: CML Logistic Ring experiments completed.")
