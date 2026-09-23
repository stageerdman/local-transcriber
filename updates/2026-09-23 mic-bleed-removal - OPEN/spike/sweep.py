"""Sweep alpha (attenuation), tau (delay), overlap fraction, and many clean
seeds. Reports where detection succeeds/breaks, direction-error rate, and the
false-positive rate on clean pairs. Used to pick tier thresholds.
"""

import numpy as np
from synth import make_pair, SR
import detect


def run_clean_fp(n_seeds=60):
    """False-positive study: clean/independent pairs. What rho do they reach?"""
    rhos, wrong_dir_calls = [], 0
    tiers = {"auto-clean": 0, "ask": 0, "refuse": 0}
    for s in range(n_seeds):
        # vary overlap so clean pairs sometimes overlap in time (still independent)
        on = np.random.default_rng(1000 + s).uniform(0.4, 0.95)
        A, B, _, _ = make_pair(seed=1000 + s, alpha_ba=0.0, alpha_ab=0.0,
                               on_frac_a=on, on_frac_b=on)
        m = detect.analyze_pair(A, B)
        verdict, tier, _ = detect.decide(m)
        rhos.append(m["rho"])
        tiers[tier] += 1
        if verdict.startswith("bleed") and "?" not in verdict:
            wrong_dir_calls += 1  # confident bleed on a clean pair = false positive
    rhos = np.array(rhos)
    print("\n--- CLEAN-PAIR FALSE-POSITIVE STUDY (independent voices) ---")
    print(f"  n={n_seeds}  rho: mean={rhos.mean():.3f} p50={np.median(rhos):.3f} "
          f"p95={np.percentile(rhos,95):.3f} max={rhos.max():.3f}")
    print(f"  tiers assigned: {tiers}")
    print(f"  CONFIDENT-bleed false positives (auto-clean gate on clean pair): "
          f"{wrong_dir_calls}/{n_seeds}")
    return rhos


def run_alpha_tau_sweep():
    """B->A only. Grid over alpha and tau. Record detected rho, direction
    correctness, lag error, and tier."""
    alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]
    taus_ms = [1, 2, 5, 8, 12, 20]
    print("\n--- ALPHA x TAU SWEEP (B->A only; want dir=B->A, correct lag) ---")
    header = "alpha\\tau |" + "".join(f"{t:>6}ms" for t in taus_ms)
    print(header)
    print("  (cell = rho / tier-code; tier: C=auto-clean-bleed A=ask R=refuse "
          ". =clean/miss ; ! = DIRECTION ERROR)")
    tier_code = {"auto-clean": "C", "ask": "A", "refuse": "R"}
    for a in alphas:
        row = [f"  {a:.2f}   |"]
        for tms in taus_ms:
            tau = int(tms / 1000 * SR)
            # average over a few seeds for stability
            rs, dir_err, tiers, lagerr = [], 0, [], []
            for sd in range(4):
                A, B, _, _ = make_pair(seed=500 + sd, alpha_ba=a, tau_ba=tau)
                m = detect.analyze_pair(A, B)
                verdict, tier, _ = detect.decide(m)
                rs.append(m["rho"])
                tiers.append(tier)
                lagerr.append(abs(m["lag"] - tau))
                # a "detected bleed" that is not B->A is a direction error
                if (tier != "auto-clean" or verdict.startswith("bleed")) and \
                   m["direction"] not in ("B->A",) and m["rho"] >= detect.RHO_CLEAN:
                    dir_err += 1
            rho_m = np.mean(rs)
            # majority tier
            from collections import Counter
            tc = Counter(tiers).most_common(1)[0][0]
            code = tier_code[tc]
            if rho_m < detect.RHO_CLEAN:
                code = "."
            flag = "!" if dir_err >= 2 else " "
            row.append(f"{rho_m:4.2f}{code}{flag}")
        print("".join(row))


def run_overlap_sweep():
    """Fixed bleed, vary how much the two voices overlap in TIME (both active
    simultaneously). High overlap should make gating harder / push to ask."""
    print("\n--- OVERLAP SWEEP (B->A, alpha=0.3, tau=8ms; vary on_frac of both) ---")
    tau = int(0.008 * SR)
    for on in [0.3, 0.5, 0.7, 0.9]:
        A, B, _, _ = make_pair(seed=42, alpha_ba=0.3, tau_ba=tau,
                               on_frac_a=on, on_frac_b=on)
        m = detect.analyze_pair(A, B)
        verdict, tier, _ = detect.decide(m)
        print(f"  on_frac={on:.1f}: rho={m['rho']:.3f} dir={m['direction']} "
              f"cons={m['lag_consistency']:.2f} sym={m.get('side_symmetry',0):.2f} "
              f"-> {verdict} [{tier}]")


def run_mutual_asymmetry():
    """Mutual bleed with DIFFERENT strengths — can we still refuse, or do we
    mislabel the louder direction as clean-gateable?"""
    print("\n--- MUTUAL w/ ASYMMETRIC STRENGTH (alpha_ba vs alpha_ab) ---")
    tba, tab = int(0.007 * SR), int(0.004 * SR)
    for aba, aab in [(0.4, 0.1), (0.4, 0.2), (0.4, 0.3), (0.5, 0.5), (0.6, 0.4)]:
        A, B, _, _ = make_pair(seed=77, alpha_ba=aba, tau_ba=tba,
                               alpha_ab=aab, tau_ab=tab,
                               on_frac_a=0.9, on_frac_b=0.9)
        m = detect.analyze_pair(A, B)
        verdict, tier, _ = detect.decide(m)
        print(f"  a_ba={aba} a_ab={aab}: rho={m['rho']:.3f} dir={m['direction']} "
              f"sym={m.get('side_symmetry',0):.2f} -> {verdict} [{tier}]")


if __name__ == "__main__":
    print("MIC-BLEED DETECTABILITY SWEEP")
    run_clean_fp()
    run_alpha_tau_sweep()
    run_overlap_sweep()
    run_mutual_asymmetry()
