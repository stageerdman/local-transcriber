"""Run the four required scenarios and print measured numbers + verdicts."""

import numpy as np
from synth import make_pair, make_triple, SR
import detect


def show(title, A, B, **kw):
    m = detect.analyze_pair(A, B, **kw)
    verdict, tier, reason = detect.decide(m)
    mask = detect.gating_mask(m)
    gate_frac = mask.mean() if len(mask) else 0.0
    print(f"\n=== {title} ===")
    print(f"  rho={m['rho']:.3f}  lag={m['lag']} smp ({m['lag_ms']:.1f} ms)  "
          f"dir={m['direction']}  consistency={m['lag_consistency']:.2f}  "
          f"side_sym={m.get('side_symmetry', 0):.2f}  n_active={m['n_active']}")
    print(f"  VERDICT: {verdict:16s} TIER: {tier:11s}  ({reason})")
    print(f"  gating windows flagged: {gate_frac*100:.0f}% of all windows")
    return m, verdict, tier


def main():
    print("Mic-bleed detectability spike — required scenarios")
    print(f"sample rate {SR} Hz, ~6 s clips")

    # (a) clean A, clean B -> expect NO bleed both directions
    A, B, _, _ = make_pair(seed=1, alpha_ba=0.0, alpha_ab=0.0)
    show("(a) clean A, clean B", A, B)

    # (b) B bleeds into A only. alpha 0.3, tau 8 ms
    tau = int(0.008 * SR)
    A, B, _, _ = make_pair(seed=2, alpha_ba=0.3, tau_ba=tau)
    show(f"(b) B->A only  (alpha=0.30, tau=8ms={tau}smp)", A, B)

    # a fainter one at edge of spec
    tau2 = int(0.003 * SR)
    A, B, _, _ = make_pair(seed=3, alpha_ba=0.15, tau_ba=tau2)
    show(f"(b') faint B->A (alpha=0.15, tau=3ms={tau2}smp)", A, B)

    # (c) mutual heavy overlap: both loud throughout, strong bleed each way
    tau_ba = int(0.007 * SR)
    tau_ab = int(0.005 * SR)
    A, B, _, _ = make_pair(seed=4, alpha_ba=0.6, tau_ba=tau_ba,
                           alpha_ab=0.6, tau_ab=tau_ab,
                           on_frac_a=0.95, on_frac_b=0.95)
    show(f"(c) mutual heavy (alpha=0.6 both ways)", A, B)

    # (d) 3-track: p3 clean, p2=p2+bleed(p3), p1 clean
    t = make_triple(seed=7)
    print("\n\n### (d) 3-track scenario: P1 clean, P2 = P2 + bleed(P3), P3 clean")
    print(f"    (ground truth: P3 -> P2, tau={t['tau_p3_into_p2']} smp = 6 ms)")
    tracks = {"P1": t["p1"], "P2": t["p2"], "P3": t["p3"]}
    names = list(tracks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            # analyze_pair(A,B): direction "B->A" means the SECOND arg bleeds
            # into the FIRST. Report with explicit names.
            m = detect.analyze_pair(tracks[ni], tracks[nj])
            verdict, tier, reason = detect.decide(m)
            # translate direction to real names
            d = m["direction"]
            if d == "B->A":
                human = f"{nj} -> {ni}"
            elif d == "A->B":
                human = f"{ni} -> {nj}"
            else:
                human = d
            print(f"\n  pair {ni},{nj}: rho={m['rho']:.3f} lag={m['lag_ms']:.1f}ms "
                  f"dir={human} cons={m['lag_consistency']:.2f} sym={m.get('side_symmetry',0):.2f}")
            print(f"      VERDICT {verdict} | TIER {tier} | {reason}")


if __name__ == "__main__":
    main()
