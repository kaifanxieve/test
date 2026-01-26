# =========================================================
# BREAK-TEST HARNESS: try to "break" the claim via code
# Systems: ECA / RD / CML (minimal versions)
# Attacks: threshold sweep, coarse-grain operator swap, scan order swap
# =========================================================

import numpy as np
import math
from collections import Counter

# -----------------------------
# 0) Reproducibility
# -----------------------------
BASE_SEED = 20260126
def rng_for_trial(trial_id: int):
    return np.random.default_rng(BASE_SEED + int(trial_id))

# -----------------------------
# 1) Complexity: Normalised LZ (binary / multi-symbol OK)
# -----------------------------
def lempel_ziv_complexity(seq):
    s = list(seq)
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
    # n/log2(n) normalisation (as in your scripts)
    return c / (n / math.log2(n))

# -----------------------------
# 2) Coarse-graining operators
# -----------------------------
def coarse_grain_row(row, sigma, mode="majority"):
    """
    row: 1D array of ints (0/1) or small ints
    mode: majority / OR / AND / parity
    """
    N = len(row)
    if N % sigma != 0:
        raise ValueError("sigma must divide N exactly")
    blocks = N // sigma
    out = np.zeros(blocks, dtype=int)
    for b in range(blocks):
        block = row[b*sigma:(b+1)*sigma]
        if mode == "majority":
            out[b] = 1 if np.mean(block) >= 0.5 else 0
        elif mode == "OR":
            out[b] = 1 if np.any(block != 0) else 0
        elif mode == "AND":
            out[b] = 1 if np.all(block != 0) else 0
        elif mode == "parity":
            out[b] = int(np.sum(block) % 2)
        else:
            raise ValueError("unknown mode")
    return out

def coarse_grain_history(hist, sigma, mode="majority"):
    return np.array([coarse_grain_row(row, sigma, mode=mode) for row in hist], dtype=int)

# -----------------------------
# 3) Scan order
# -----------------------------
def flatten_hist(cg_hist, scan="time_major"):
    # cg_hist shape: (T, N')
    if scan == "time_major":
        return cg_hist.reshape(-1)
    elif scan == "space_major":
        return cg_hist.T.reshape(-1)
    else:
        raise ValueError("unknown scan")

# -----------------------------
# 4) "Window / flip" detectors
# -----------------------------
def has_intermediate_peak(sigmas, C, min_prominence=0.02):
    """
    Very simple: peak exists if max is not at endpoints and exceeds endpoints by min_prominence.
    """
    C = np.asarray(C, dtype=float)
    i_max = int(np.argmax(C))
    if i_max == 0 or i_max == len(C)-1:
        return False, i_max
    endpoints = max(C[0], C[-1])
    return (C[i_max] - endpoints) >= min_prominence, i_max

def ranking_flip(sigmas, C_by_label):
    """
    C_by_label: dict label -> list C(sigma_i)
    Returns True if ordering changes for any pair across sigma.
    """
    labels = list(C_by_label.keys())
    ranks = []
    for i in range(len(sigmas)):
        ordering = sorted(labels, key=lambda L: C_by_label[L][i], reverse=True)
        ranks.append(ordering)
    return any(ranks[i] != ranks[0] for i in range(1, len(ranks))), ranks

# =========================================================
# SYSTEM A) ECA (binary already)
# =========================================================
def rule_to_table(rule):
    bits = np.binary_repr(rule, width=8)
    return np.array([int(b) for b in bits])[::-1]

def evolve_ca(state, table, rng, p_noise=0.0):
    N = len(state)
    new_state = np.zeros_like(state)
    for i in range(N):
        l = state[(i-1) % N]
        c = state[i]
        r = state[(i+1) % N]
        idx = (l << 2) | (c << 1) | r
        new_state[i] = table[idx]
    if p_noise > 0:
        noise = rng.random(N) < p_noise
        new_state = np.logical_xor(new_state, noise).astype(int)
    return new_state

def run_eca(rule, N, T, rng, init="random", p_noise=0.0):
    table = rule_to_table(rule)
    if init == "random":
        state = rng.integers(0, 2, size=N)
    else:
        state = np.zeros(N, dtype=int); state[N//2] = 1
    hist = np.zeros((T, N), dtype=int)
    for t in range(T):
        hist[t] = state
        state = evolve_ca(state, table, rng, p_noise=p_noise)
    return hist

# =========================================================
# SYSTEM B) RD (Gray–Scott 1D) -> binary by threshold
# =========================================================
def laplacian_1d(x):
    return np.roll(x, 1) + np.roll(x, -1) - 2*x

def run_rd(N, T, rng, Du=0.16, Dv=0.08, F=0.035, k=0.065, p_noise=0.0):
    u = np.ones(N)
    v = np.zeros(N)
    v[N//2-5:N//2+5] = 1.0
    u -= v
    hist_v = np.zeros((T, N), dtype=float)
    for t in range(T):
        hist_v[t] = v
        Lu = laplacian_1d(u)
        Lv = laplacian_1d(v)
        uvv = u * v * v
        u += Du * Lu - uvv + F * (1 - u)
        v += Dv * Lv + uvv - (F + k) * v
        if p_noise > 0:
            v += p_noise * rng.normal(0, 0.1, size=N)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
    return hist_v

def rd_to_binary(hist_v, thr=0.5):
    return (hist_v >= thr).astype(int)

# =========================================================
# SYSTEM C) CML Logistic ring -> binary by threshold
# =========================================================
def logistic_map(x, r):
    return r * x * (1.0 - x)

def run_cml(N, T, rng, r=3.9, eps=0.25, p_noise=0.0, noise_sigma=0.02):
    x = rng.random(N)
    hist_x = np.zeros((T, N), dtype=float)
    for t in range(T):
        hist_x[t] = x
        fx = logistic_map(x, r)
        fx_l = np.roll(fx, 1)
        fx_r = np.roll(fx, -1)
        x_next = (1.0 - eps) * fx + (eps * 0.5) * (fx_l + fx_r)
        if p_noise > 0:
            mask = rng.random(N) < p_noise
            x_next = x_next + mask * rng.normal(0.0, noise_sigma, size=N)
        x = np.clip(x_next, 0.0, 1.0)
    return hist_x

def cml_to_binary(hist_x, thr=0.5):
    return (hist_x >= thr).astype(int)

# =========================================================
# CORE: compute C(sigma) curve for a given binary history
# =========================================================
def multiscale_C_from_binary(hist_bin, sigmas, cg_mode="majority", scan="time_major"):
    C = []
    for sigma in sigmas:
        cg = coarse_grain_history(hist_bin, sigma, mode=cg_mode)
        seq = flatten_hist(cg, scan=scan)
        C.append(lz_normalised(seq))
    return C

# =========================================================
#  RUN "BREAK TESTS"
# =========================================================
def break_test():
    N = 512
    T = 700
    TRANSIENT = 150
    REPEATS = 10  # keep moderate; increase if needed
    SIGMAS = [1,2,4,8,16,32,64]

    THRS = [0.3, 0.4, 0.5, 0.6, 0.7]          # attack #1
    CG_MODES = ["majority", "OR", "AND", "parity"]  # attack #2
    SCANS = ["time_major", "space_major"]      # attack #3

    # --- ECA rules for ranking/flip test
    RULES = [30,54,90,110]

    # --- CML r list for ranking/flip test
    R_LIST = [3.5, 3.7, 3.9, 4.0]

    # --- RD parameter list for ranking/flip-style test (treat as "labels")
    RD_LIST = [
        ("F0.030_k0.062", dict(F=0.030, k=0.062)),
        ("F0.035_k0.065", dict(F=0.035, k=0.065)),
        ("F0.040_k0.060", dict(F=0.040, k=0.060)),
        ("F0.022_k0.051", dict(F=0.022, k=0.051)),
    ]

    def avg_curve(curves):
        return np.mean(np.array(curves, dtype=float), axis=0).tolist()

    results = []

    # -------------- ECA: cg_mode + scan sensitivity --------------
    for cg_mode in CG_MODES:
        for scan in SCANS:
            C_by_rule = {r: [] for r in RULES}
            for r in RULES:
                curves = []
                for rep in range(REPEATS):
                    rng = rng_for_trial(r*10000 + rep)
                    hist = run_eca(r, N, T, rng, init="random", p_noise=0.0)[TRANSIENT:]
                    curves.append(multiscale_C_from_binary(hist, SIGMAS, cg_mode=cg_mode, scan=scan))
                C_by_rule[r] = avg_curve(curves)
            flip, rank_traces = ranking_flip(SIGMAS, C_by_rule)
            # pick one rule curve to peak-check
            peak_ok, peak_idx = has_intermediate_peak(SIGMAS, C_by_rule[110])
            results.append(("ECA", f"cg={cg_mode},scan={scan}", {"flip": flip, "peak(rule110)": peak_ok, "peak_idx": peak_idx, "rank0": rank_traces[0], "rank_last": rank_traces[-1]}))

    # -------------- RD: threshold + cg_mode + scan --------------
    for thr in THRS:
        for cg_mode in CG_MODES:
            for scan in SCANS:
                C_by_param = {lab: [] for lab,_ in RD_LIST}
                for lab, pars in RD_LIST:
                    curves = []
                    for rep in range(REPEATS):
                        rng = rng_for_trial(hash((lab, rep)) % 10_000_000)
                        hist_v = run_rd(N, T, rng, **pars, p_noise=0.0)[TRANSIENT:]
                        hist_bin = rd_to_binary(hist_v, thr=thr)
                        curves.append(multiscale_C_from_binary(hist_bin, SIGMAS, cg_mode=cg_mode, scan=scan))
                    C_by_param[lab] = avg_curve(curves)

                flip, rank_traces = ranking_flip(SIGMAS, C_by_param)
                # peak check on the default param
                peak_ok, peak_idx = has_intermediate_peak(SIGMAS, C_by_param["F0.035_k0.065"])
                results.append(("RD", f"thr={thr},cg={cg_mode},scan={scan}", {"flip": flip, "peak(default)": peak_ok, "peak_idx": peak_idx, "rank0": rank_traces[0], "rank_last": rank_traces[-1]}))

    # -------------- CML: threshold + cg_mode + scan --------------
    for thr in THRS:
        for cg_mode in CG_MODES:
            for scan in SCANS:
                C_by_r = {r: [] for r in R_LIST}
                for r in R_LIST:
                    curves = []
                    for rep in range(REPEATS):
                        rng = rng_for_trial(int(r*100000) + rep)
                        hist_x = run_cml(N, T, rng, r=r, eps=0.25, p_noise=0.0)[TRANSIENT:]
                        hist_bin = cml_to_binary(hist_x, thr=thr)
                        curves.append(multiscale_C_from_binary(hist_bin, SIGMAS, cg_mode=cg_mode, scan=scan))
                    C_by_r[r] = avg_curve(curves)

                flip, rank_traces = ranking_flip(SIGMAS, C_by_r)
                peak_ok, peak_idx = has_intermediate_peak(SIGMAS, C_by_r[3.9])
                results.append(("CML", f"thr={thr},cg={cg_mode},scan={scan}", {"flip": flip, "peak(r=3.9)": peak_ok, "peak_idx": peak_idx, "rank0": rank_traces[0], "rank_last": rank_traces[-1]}))

    # Print only "successful attacks": cases where RD/CML shows peak OR flip, or where ECA loses them
    print("\n=== BREAK SUCCESSES (potential counterexamples / artifact evidence) ===")
    for sys, setting, info in results:
        if sys in ("RD","CML"):
            if info["flip"] or info.get("peak(default)", False) or info.get("peak(r=3.9)", False):
                print(f"[{sys}] {setting} -> {info}")
        if sys == "ECA":
            # if ECA loses both peak & flip under some pipeline, that's artifact alarm
            if (not info["flip"]) and (not info["peak(rule110)"]):
                print(f"[ECA-ALARM] {setting} -> {info}")

    return results

if __name__ == "__main__":
    break_test()
