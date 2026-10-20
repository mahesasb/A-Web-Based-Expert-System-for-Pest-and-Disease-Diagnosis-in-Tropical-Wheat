#!/usr/bin/env python3
"""
Reference implementation and simulation-based evaluation of the Forward-Chaining +
Certainty-Factor (FC+CF) expert system for tropical-wheat pest/disease diagnosis.

Everything reported in the revised manuscript that is a *computed* number is produced here
(fixed seeds -> fully reproducible).  Run:  python3 wheat_es_experiments.py

Outputs: results.json, fig_accuracy_noise.png, fig_confusion.png, fig_calibration.png
"""
import itertools
import json
import math
import random

import numpy as np

SEED = 20241019
EPS = 1e-9

# ----------------------------------------------------------------------------------------
# 1. Knowledge base (Table 6 of the manuscript): rule (disease, symptom) -> (MB, MD)
# ----------------------------------------------------------------------------------------
DISEASES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
DNAMES = {
    "P1": "Leaf Rust", "P2": "Leaf Blight", "P3": "Fusarium Crown/Root Rot",
    "P4": "Stem Rust", "P5": "Powdery Mildew", "P6": "Fusarium Head Blight",
    "P7": "Aphids", "P8": "Spodoptera litura",
}
SYMPTOMS = [f"G{i}" for i in range(1, 14)]
RULES = {
    "P1": {"G1": (0.8, 0.2), "G2": (1.0, 0.0)},
    "P2": {"G4": (0.8, 0.2), "G9": (0.6, 0.2)},
    "P3": {"G3": (0.6, 0.2), "G4": (0.6, 0.2), "G5": (1.0, 0.0)},
    "P4": {"G3": (0.6, 0.2), "G6": (1.0, 0.0)},
    "P5": {"G7": (1.0, 0.0), "G9": (0.6, 0.2)},
    "P6": {"G12": (0.8, 0.2), "G13": (1.0, 0.0)},
    "P7": {"G8": (1.0, 0.0), "G9": (0.8, 0.2)},
    "P8": {"G10": (0.8, 0.2), "G11": (1.0, 0.0)},
}
NP, NG = len(DISEASES), len(SYMPTOMS)
SIDX = {g: i for i, g in enumerate(SYMPTOMS)}
DIDX = {p: i for i, p in enumerate(DISEASES)}


def weight_matrix(rules=RULES):
    """W[p, g] = CF_expert = MB - MD (0 where no rule)."""
    W = np.zeros((NP, NG))
    for p, rs in rules.items():
        for g, (mb, md) in rs.items():
            W[DIDX[p], SIDX[g]] = round(mb - md, 10)
    return W


def link_matrix(rules=RULES):
    L = np.zeros((NP, NG))
    for p, rs in rules.items():
        for g in rs:
            L[DIDX[p], SIDX[g]] = 1.0
    return L


W0 = weight_matrix()
L0 = link_matrix()

# ----------------------------------------------------------------------------------------
# 2. Reference forward-chaining engine (explicit working memory / agenda), single session
# ----------------------------------------------------------------------------------------

def forward_chain(obs, rules=RULES, order_rng=None):
    """
    obs: {symptom: CF_user}.  Three-layer production system:
      R1  evidence rules  : IF obs(G,u) THEN evidence(P, u*CFexp)  (fires once per matching rule)
      R2  candidate rule  : IF evidence(P) > 0 THEN candidate(P)
      R3  conclusion rule : IF candidate(P) has max CF THEN conclude(P)
    Evidence for the same disease is merged incrementally: CF <- CF + cf(1-CF).
    Returns (cf_per_disease, firing_trace, conclusion).
    """
    agenda = [(g, p) for g in obs for p in DISEASES if g in rules[p]]
    if order_rng is not None:
        order_rng.shuffle(agenda)
    cf = {}
    trace = []
    while agenda:                      # run until no rule can fire
        g, p = agenda.pop(0)
        mb, md = rules[p][g]
        c = obs[g] * round(mb - md, 10)
        old = cf.get(p, 0.0)
        cf[p] = old + c * (1.0 - old)
        trace.append((g, p, round(c, 6), round(cf[p], 6)))
    cands = {p: v for p, v in cf.items() if v > EPS}
    conclusion = None
    if cands:
        top = max(cands.values())
        conclusion = sorted([p for p, v in cands.items() if abs(v - top) < 1e-9])
    return cf, trace, conclusion


# ----------------------------------------------------------------------------------------
# 3. Vectorised scoring functions (proposed + baselines)   U: (n, NG) user CFs, 0 = not reported
# ----------------------------------------------------------------------------------------

def score_proposed(U, W=W0):
    return 1.0 - np.prod(1.0 - U[:, None, :] * W[None, :, :], axis=2)


def score_cf_user_only(U, L=L0):
    return 1.0 - np.prod(1.0 - U[:, None, :] * L[None, :, :], axis=2)


def score_count(U, L=L0):
    return ((U > 0)[:, None, :] * L[None, :, :]).sum(axis=2)


def score_weighted_coverage(U, W=W0):
    num = (U[:, None, :] * W[None, :, :]).sum(axis=2)
    den = W.sum(axis=1)[None, :]
    return num / den


METHODS = {
    "Rule matching (count)": score_count,
    "CF, user only": score_cf_user_only,
    "Weighted coverage": score_weighted_coverage,
    "Proposed FC+CF": score_proposed,
}

# ----------------------------------------------------------------------------------------
# 4. Metrics with exact random-tie-breaking expectation
# ----------------------------------------------------------------------------------------

def rank_stats(S, y, ks=(1, 2, 3)):
    n = len(y)
    st = S[np.arange(n), y]
    higher = (S > st[:, None] + EPS).sum(axis=1)
    tied = (np.abs(S - st[:, None]) <= EPS).sum(axis=1)
    ok = st > EPS                       # disease must be retrieved (score > 0)
    out = {}
    for k in ks:
        out[f"top{k}"] = np.where(ok, np.clip((k - higher) / tied, 0, 1), 0.0)
    mrr = np.zeros(n)
    for i in range(n):
        if ok[i]:
            h, t = higher[i], tied[i]
            mrr[i] = np.mean([1.0 / r for r in range(h + 1, h + t + 1)])
    out["mrr"] = mrr
    mx = S.max(axis=1)
    ntop = (np.abs(S - mx[:, None]) <= EPS).sum(axis=1)
    out["tie_top1"] = ((ntop > 1) & (mx > EPS)).astype(float)
    return out


def mean_ci(a):
    m = float(np.mean(a))
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, 1.96 * se

# ----------------------------------------------------------------------------------------
# 5. Session simulator
# ----------------------------------------------------------------------------------------
CONF_MODELS = {
    # farmer is more confident about real symptoms than about mistaken ones
    "calibrated": dict(
        true_levels=[0.4, 0.6, 0.8, 1.0], true_p=[0.10, 0.30, 0.40, 0.20],
        false_levels=[0.2, 0.4, 0.6], false_p=[0.30, 0.40, 0.30]),
    # worst case: confidence carries no information about correctness
    "uninformative": dict(
        true_levels=[0.2, 0.4, 0.6, 0.8, 1.0], true_p=[0.10, 0.20, 0.30, 0.25, 0.15],
        false_levels=[0.2, 0.4, 0.6, 0.8, 1.0], false_p=[0.10, 0.20, 0.30, 0.25, 0.15]),
}


def simulate(n_per_disease, p_obs, p_fp, conf, rng, rules=RULES):
    cm = CONF_MODELS[conf]
    U, y = [], []
    for p in DISEASES:
        true_syms = [SIDX[g] for g in rules[p]]
        false_syms = [i for i in range(NG) if i not in true_syms]
        for _ in range(n_per_disease):
            u = np.zeros(NG)
            rep = [i for i in true_syms if rng.random() < p_obs]
            if not rep:
                rep = [int(rng.choice(true_syms))]
            for i in rep:
                u[i] = rng.choice(cm["true_levels"], p=cm["true_p"])
            for i in false_syms:
                if rng.random() < p_fp:
                    u[i] = rng.choice(cm["false_levels"], p=cm["false_p"])
            U.append(u)
            y.append(DIDX[p])
    return np.array(U), np.array(y)


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED}

    # ---------------- E0: verification of the engine -------------------------------------
    worked = {"G1": 0.6, "G2": 0.6}
    cf, trace, concl = forward_chain(worked)
    res["E0_worked_example"] = {"cf_P1": cf["P1"], "trace": trace, "conclusion": concl}
    # extended competing-hypotheses example
    ex2 = {"G1": 0.6, "G2": 0.6, "G9": 0.4}
    cf2, trace2, concl2 = forward_chain(ex2)
    res["E0_competing_example"] = {
        "obs": ex2, "cf": {p: round(v, 4) for p, v in sorted(cf2.items())},
        "trace": trace2, "conclusion": concl2}

    # property tests on random sessions
    prng = random.Random(SEED)
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_test = 5000
    order_ok = bound_ok = mono_ok = closed_ok = 0
    for _ in range(n_test):
        k = prng.randint(1, NG)
        syms = prng.sample(SYMPTOMS, k)
        obs = {g: prng.choice(levels) for g in syms}
        base, _, _ = forward_chain(obs)
        # (a) order independence
        shuffled, _, _ = forward_chain(obs, order_rng=random.Random(prng.random()))
        if all(abs(base.get(p, 0) - shuffled.get(p, 0)) < 1e-12 for p in DISEASES):
            order_ok += 1
        # (b) bounded in [0,1]
        if all(-1e-12 <= v <= 1 + 1e-12 for v in base.values()):
            bound_ok += 1
        # (c) closed form (noisy-OR) agreement
        okc = True
        for p in DISEASES:
            prod = 1.0
            for g, u in obs.items():
                if g in RULES[p]:
                    mb, md = RULES[p][g]
                    prod *= 1 - u * round(mb - md, 10)
            okc &= abs((1 - prod) - base.get(p, 0.0)) < 1e-12
        closed_ok += okc
        # (d) monotonic: adding one extra symptom never lowers any disease CF
        extra = [g for g in SYMPTOMS if g not in obs]
        if extra:
            obs2 = dict(obs)
            obs2[prng.choice(extra)] = prng.choice(levels)
            more, _, _ = forward_chain(obs2)
            mono_ok += all(more.get(p, 0) >= base.get(p, 0) - 1e-12 for p in DISEASES)
        else:
            mono_ok += 1
    res["E0_properties"] = dict(n=n_test, order_independence=order_ok, bounded=bound_ok,
                                closed_form=closed_ok, monotonic=mono_ok)

    # vectorised vs reference engine consistency
    U_chk, y_chk = simulate(25, 0.8, 0.1, "calibrated", rng)
    S_vec = score_proposed(U_chk)
    mism = 0
    for i in range(len(U_chk)):
        obs = {SYMPTOMS[j]: float(U_chk[i, j]) for j in range(NG) if U_chk[i, j] > 0}
        c, _, _ = forward_chain(obs)
        for p in DISEASES:
            mism += abs(c.get(p, 0.0) - S_vec[i, DIDX[p]]) > 1e-9
    res["E0_vectorised_check"] = dict(sessions=len(U_chk), mismatches=int(mism))

    # ---------------- E1: knowledge-base ambiguity analysis -------------------------------
    sym_deg = {g: int(L0[:, SIDX[g]].sum()) for g in SYMPTOMS}
    res["E1_symptom_degree"] = sym_deg
    single = {}
    for g in SYMPTOMS:
        row = {}
        for u in (1.0, 0.6):
            U = np.zeros((1, NG)); U[0, SIDX[g]] = u
            s = score_proposed(U)[0]
            order = sorted([(DISEASES[i], round(float(s[i]), 3)) for i in range(NP) if s[i] > EPS],
                           key=lambda t: -t[1])
            row[str(u)] = order
        single[g] = row
    res["E1_single_symptom"] = single
    n_amb = sum(1 for g in SYMPTOMS
                if sum(1 for _, v in single[g]["1.0"] if abs(v - single[g]["1.0"][0][1]) < 1e-9) > 1)
    res["E1_single_ambiguous_top1"] = n_amb
    # exclusive symptoms per disease
    excl = {p: [g for g in RULES[p] if sym_deg[g] == 1] for p in DISEASES}
    res["E1_exclusive"] = excl
    # exhaustive enumeration of all symptom combinations of size 1..3 (plausible farmer reports)
    enum = {}
    for u in (1.0, 0.6):
        subsets = []
        for k in (1, 2, 3):
            for comb in itertools.combinations(range(NG), k):
                v = np.zeros(NG)
                for j in comb:
                    v[j] = u
                subsets.append(v)
        Uall = np.array(subsets)
        S = score_proposed(Uall)
        mx = S.max(axis=1)
        ntop = (np.abs(S - mx[:, None]) <= EPS).sum(axis=1)
        enum[str(u)] = dict(
            n=len(Uall), unique_top1=float((ntop == 1).mean()),
            top_cf_ge_08=float((mx >= 0.8).mean()), top_cf_ge_06=float((mx >= 0.6).mean()),
            mean_candidates=float(((S > EPS).sum(axis=1)).mean()))
    res["E1_enumeration"] = enum

    # ---------------- E2: Monte-Carlo consultation simulation -----------------------------
    N_PER = 2000
    grid = {}
    for conf in CONF_MODELS:
        for p_obs in (1.0, 0.8, 0.6):
            for p_fp in (0.0, 0.05, 0.10, 0.20):
                U, y = simulate(N_PER, p_obs, p_fp, conf, rng)
                cell = {}
                for name, fn in METHODS.items():
                    st = rank_stats(fn(U), y)
                    cell[name] = {k: mean_ci(v) for k, v in st.items()}
                grid[f"{conf}|{p_obs}|{p_fp}"] = cell
    res["E2_grid"] = grid
    res["E2_n_per_cell"] = N_PER * NP

    # default condition: detailed analysis
    DEF = dict(p_obs=0.8, p_fp=0.05, conf="calibrated")
    res["default_condition"] = DEF
    U, y = simulate(N_PER, DEF["p_obs"], DEF["p_fp"], DEF["conf"], rng)
    S = score_proposed(U)
    # per-disease top-1 / top-3 and confusion (expected counts under random tie-break)
    st = rank_stats(S, y)
    per_dis = {}
    for p in DISEASES:
        m = (y == DIDX[p])
        per_dis[p] = dict(top1=float(st["top1"][m].mean()), top3=float(st["top3"][m].mean()))
    res["E2_per_disease"] = per_dis
    conf_m = np.zeros((NP, NP))
    for i in range(len(y)):
        s = S[i]
        mx = s.max()
        if mx <= EPS:
            continue
        top = np.where(np.abs(s - mx) <= EPS)[0]
        for t in top:
            conf_m[y[i], t] += 1.0 / len(top)
    row_sum = np.array([(y == k).sum() for k in range(NP)], dtype=float)
    conf_norm = conf_m / row_sum[:, None]
    res["E2_confusion"] = conf_norm.tolist()

    # significance of proposed vs each baseline (paired difference of per-session top-1 credit)
    base_top1 = rank_stats(score_proposed(U), y)["top1"]
    paired = {}
    for name, fn in METHODS.items():
        if name == "Proposed FC+CF":
            continue
        d = base_top1 - rank_stats(fn(U), y)["top1"]
        paired[name] = dict(diff=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / math.sqrt(len(d))))
    res["E2_paired_top1_diff"] = paired

    # ---------------- E3: robustness to perturbation of expert parameters -----------------
    sens = {}
    nominal = rank_stats(score_proposed(U), y)["top1"].mean()
    for delta in (0.05, 0.10, 0.20):
        accs = []
        for _ in range(50):
            Wp = np.zeros_like(W0)
            for p, rs in RULES.items():
                for g, (mb, md) in rs.items():
                    mbp = np.clip(mb + rng.uniform(-delta, delta), 0, 1)
                    mdp = np.clip(md + rng.uniform(-delta, delta), 0, 1)
                    Wp[DIDX[p], SIDX[g]] = max(mbp - mdp, 0.0)
            accs.append(rank_stats(score_proposed(U, Wp), y)["top1"].mean())
        sens[str(delta)] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)),
                                min=float(np.min(accs)), max=float(np.max(accs)))
    res["E3_sensitivity"] = dict(nominal=float(nominal), runs=50, deltas=sens)

    # ---------------- E4: calibration of CF_top and selective prediction ------------------
    top_cf = S.max(axis=1)
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    cal = []
    for lo, hi in bands:
        m = (top_cf >= lo) & (top_cf < hi)
        if m.sum() == 0:
            cal.append(dict(lo=lo, hi=min(hi, 1.0), share=0.0, top1=None, n=0))
            continue
        cal.append(dict(lo=lo, hi=min(hi, 1.0), share=float(m.mean()),
                        top1=float(st["top1"][m].mean()), n=int(m.sum())))
    res["E4_calibration"] = cal
    pol = []
    for tau in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = top_cf >= max(tau, EPS)
        pol.append(dict(tau=tau, coverage=float(m.mean()),
                        top1=float(st["top1"][m].mean()) if m.any() else None,
                        top3_shortlist=float(st["top3"][m].mean()) if m.any() else None))
    res["E4_policy_default"] = pol
    # hard condition
    Uh, yh = simulate(N_PER, 0.6, 0.10, "calibrated", rng)
    Sh = score_proposed(Uh)
    sth = rank_stats(Sh, yh)
    tch = Sh.max(axis=1)
    polh = []
    for tau in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = tch >= max(tau, EPS)
        polh.append(dict(tau=tau, coverage=float(m.mean()),
                         top1=float(sth["top1"][m].mean()) if m.any() else None))
    res["E4_policy_hard"] = polh
    res["E4_hard_condition"] = dict(p_obs=0.6, p_fp=0.10, conf="calibrated")
    # continuous curve for figure
    taus = np.linspace(0, 0.95, 20)
    curve = []
    for tau in taus:
        m = top_cf >= max(tau, EPS)
        curve.append((float(tau), float(m.mean()), float(st["top1"][m].mean()) if m.any() else float("nan")))
    res["E4_curve"] = curve

    with open("results.json", "w") as f:
        json.dump(res, f, indent=1, default=float)

    make_figures(res)
    print("done")


# ----------------------------------------------------------------------------------------
# 6. Figures (grayscale-friendly, serif font to match the journal template)
# ----------------------------------------------------------------------------------------

def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.6})
    styles = {
        "Rule matching (count)": dict(color="0.55", marker="s", ls=":"),
        "CF, user only": dict(color="0.35", marker="^", ls="--"),
        "Weighted coverage": dict(color="0.6", marker="D", ls="-."),
        "Proposed FC+CF": dict(color="black", marker="o", ls="-"),
    }
    # Fig A: accuracy vs false-symptom rate
    fpr = [0.0, 0.05, 0.10, 0.20]
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.3), sharey=True)
    for ax, conf in zip(axes, ["calibrated", "uninformative"]):
        for name, st in styles.items():
            ys = [res["E2_grid"][f"{conf}|0.8|{p}"][name]["top1"][0] * 100 for p in fpr]
            ax.plot([p * 100 for p in fpr], ys, label=name, lw=1.1, ms=3.5, **st)
        ax.set_title(f"({'a' if conf=='calibrated' else 'b'}) {conf} confidence", fontsize=8)
        ax.set_xlabel("False-symptom rate (%)")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xticks([0, 5, 10, 20])
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend(fontsize=6, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig("fig_accuracy_noise.png", dpi=300)
    plt.close(fig)

    # Fig B: confusion matrix
    C = np.array(res["E2_confusion"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(C, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(NP)); ax.set_yticks(range(NP))
    ax.set_xticklabels(DISEASES); ax.set_yticklabels(DISEASES)
    ax.set_xlabel("Predicted (top-1)"); ax.set_ylabel("True disease")
    for i in range(NP):
        for j in range(NP):
            v = C[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 else "black")
    cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig("fig_confusion.png", dpi=300)
    plt.close(fig)

    # Fig C: calibration bars + risk-coverage curve
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.2))
    cal = [c for c in res["E4_calibration"] if c["top1"] is not None]
    labels = [f"{c['lo']:.1f}-{min(c['hi'],1.0):.1f}" for c in cal]
    axes[0].bar(range(len(cal)), [c["top1"] * 100 for c in cal], color="0.45", width=0.6)
    for i, c in enumerate(cal):
        axes[0].text(i, c["top1"] * 100 + 1.5, f"{c['share']*100:.0f}%", ha="center", fontsize=6)
    axes[0].set_xticks(range(len(cal))); axes[0].set_xticklabels(labels, fontsize=6.5)
    axes[0].set_ylim(0, 112)
    axes[0].set_xlabel("Top CF band"); axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("(a) accuracy per CF band", fontsize=8)
    cv = res["E4_curve"]
    axes[1].plot([c[0] for c in cv], [c[1] * 100 for c in cv], "k--", lw=1.1, label="Coverage")
    axes[1].plot([c[0] for c in cv], [c[2] * 100 for c in cv], "k-", lw=1.1, label="Top-1 accuracy")
    axes[1].set_xlabel("Threshold $\\tau$ on top CF"); axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("(b) selective prediction", fontsize=8)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left"); axes[1].grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig("fig_calibration.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
