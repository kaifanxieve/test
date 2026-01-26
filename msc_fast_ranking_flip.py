# =========================================================
# MSC FAST RANKING-FLIP TEST
# Goal:
#   Detect SCALE-DEPENDENT ORDERING (ranking flip),
#   NOT absolute peaks.
# =========================================================

import numpy as np
import math

# -----------------------------
# Reproducibility
# -----------------------------
BASE_SEED = 20260126
def rng(i):
    return np.random.default_rng(BASE_SEED + i)

# -----------------------------
# Lempel–Ziv complexity
# -----------------------------
def lz_complexity(seq):
    s = list(seq); n = len(s)
    i, k, l, c = 0, 1, 1, 1
    while True:
        if l + k > n:
            c += 1; break
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

def lz_norm(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    return lz_complexity(seq) / (n / math.log2(n))

# -----------------------------
# Coarse-graining
# -----------------------------
def coarse(row, sigma, mode):
    out = []
    for i in range(0, len(row), sigma):
        block = row[i:i+sigma]
        if mode == "majority":
            out.append(int(np.mean(block) >= 0.5))
        elif mode == "parity":
            out.append(int(np.sum(block) % 2))
        else:
            raise ValueError("unknown mode")
    return np.array(out, dtype=int)

def multiscale_C(history, sigmas, mode):
    C = []
    for s in sigmas:
        cg = np.array([coarse(row, s, mode) for row in history])
        C.append(lz_norm(cg.reshape(-1)))
    return C

# -----------------------------
# ECA
# -----------------------------
def run_eca(rule, N, T, rng):
    table = np.array(list(np.binary_repr(rule, 8)), dtype=int)[::-1]
    x = rng.integers(0, 2, size=N)
    hist = []
    for _ in range(T):
        hist.append(x)
        x = np.array([
            table[(x[i-1] << 2) | (x[i] << 1) | x[(i+1) % N]]
            for i in range(N)
        ])
    return np.array(hist, dtype=int)

# =========================================================
# FAST RANKING-FLIP TEST
# =========================================================
N = 512
T = 400
TRANS = 100

SIGMAS = [1, 2, 4, 8, 16, 32, 64]
RULES  = [30, 54, 90, 110]
MODES  = ["majority", "parity"]
REPEATS = 3

print("\n=== FAST RANKING-FLIP TEST START ===\n")

for mode in MODES:
    print(f"\n--- Coarse-grain mode: {mode} ---")

    # 1) Compute C(rule, sigma)
    C_by_rule = {}
    for rule in RULES:
        acc = []
        for rep in range(REPEATS):
            hist = run_eca(rule, N, T, rng(rep))[TRANS:]
            acc.append(multiscale_C(hist, SIGMAS, mode))
        C_by_rule[rule] = np.mean(acc, axis=0)

    # 2) Print ordering at each scale
    orders = []
    for i, sigma in enumerate(SIGMAS):
        order = sorted(
            RULES,
            key=lambda r: C_by_rule[r][i],
            reverse=True
        )
        orders.append(order)

        vals = [round(C_by_rule[r][i], 6) for r in order]
        print(f"σ={sigma:>2}  order={order}  C={vals}")

    # 3) Detect ranking flip
    flip = any(orders[i] != orders[0] for i in range(1, len(orders)))
    print(f"\n[mode={mode}] RANKING FLIP:", flip)

print("\n=== FAST RANKING-FLIP TEST END ===")
