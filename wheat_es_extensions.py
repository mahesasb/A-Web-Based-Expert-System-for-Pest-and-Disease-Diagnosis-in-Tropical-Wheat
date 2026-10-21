#!/usr/bin/env python3
"""
Extensions of the FC+CF expert system (simulation only):
  C  comparison with naive-Bayes and Dempster-Shafer aggregation,
  B  negative evidence (signed CF) from symptoms reported absent,
  A  adaptive interview (information-gain question selection) with a stopping rule.

Requires wheat_es_experiments.py in the same directory.  Run:  python3 wheat_es_extensions.py
Outputs: results_ext.json, fig_adaptive.png
"""
import json
import math

import numpy as np

import wheat_es_experiments as E
from wheat_es_experiments import (CONF_MODELS, DIDX, DISEASES, EPS, L0, NG, NP, RULES, SIDX, SYMPTOMS, W0,
                                  mean_ci, rank_stats, score_proposed, simulate)

SEED = 20241020
EPS_FP = 0.05                      # P(symptom reported | disease has no rule for it)
PM = np.where(L0 > 0, W0, EPS_FP)   # surrogate likelihood P(symptom present | disease)
LV5 = [0.2, 0.4, 0.6, 0.8, 1.0]
P5 = [0.10, 0.20, 0.30, 0.25, 0.15]
ABS_MODELS = {
    "calibrated": dict(correct=([0.6, 0.8, 1.0], [0.3, 0.4, 0.3]), wrong=([0.2, 0.4, 0.6], [0.4, 0.4, 0.2])),
    "uninformative": dict(correct=(LV5, P5), wrong=(LV5, P5)),
}


# ----------------------------------------------------------------------------------------------
# C. probabilistic baselines (positive reports only)
# ----------------------------------------------------------------------------------------------
def score_nb(U, eps=EPS_FP):
    """Naive Bayes with soft evidence: L = (1-u) + u*P(present|disease); uniform prior."""
    pm = np.where(L0 > 0, W0, eps)
    L = (1 - U[:, None, :]) + U[:, None, :] * pm[None, :, :]
    post = np.prod(L, axis=2)
    return post / post.sum(axis=1, keepdims=True)


def ds_session(u_row):
    """Dempster-Shafer: one simple-support function per rule (mass u*w on {P_j}); pignistic ranking."""
    m = {-1: 1.0}                                   # -1 = whole frame
    for g in np.nonzero(u_row)[0]:
        for j in range(NP):
            if L0[j, g] > 0:
                c = u_row[g] * W0[j, g]
                if c <= 0:
                    continue
                new, conflict = {}, 0.0
                for k, v in m.items():
                    if k in (-1, j):
                        new[j] = new.get(j, 0.0) + v * c
                    else:
                        conflict += v * c
                    new[k] = new.get(k, 0.0) + v * (1 - c)
                z = 1 - conflict
                if z <= 1e-12:                     # total conflict: Dempster's rule undefined -> no information
                    m = {-1: 1.0}
                    continue
                m = {k: v / z for k, v in new.items()}
    return np.array([m.get(j, 0.0) + m.get(-1, 0.0) / NP for j in range(NP)])


def score_ds(U):
    return np.array([ds_session(u) for u in U])


# ----------------------------------------------------------------------------------------------
# B. signed certainty factors
# ----------------------------------------------------------------------------------------------
def signed_scores(ans, U, M, kappa=0.5):
    """ans: +1 present / -1 absent; U: confidence; M: asked mask.  CF = (CF+ - CF-)/(1 - min(CF+, CF-))."""
    Upos = U * (ans == 1) * M
    Uneg = U * (ans == -1) * M
    cfp = 1 - np.prod(1 - Upos[:, None, :] * W0[None], axis=2)
    if kappa == 0:
        return cfp
    cfn = 1 - np.prod(1 - kappa * Uneg[:, None, :] * W0[None], axis=2)
    den = 1 - np.minimum(cfp, cfn)
    out = np.divide(cfp - cfn, den, out=np.zeros_like(cfp), where=den > EPS)
    return out


def score_nb_interview(ans, U, M):
    pres = (1 - U[:, None, :]) + U[:, None, :] * PM[None]
    absn = (1 - U[:, None, :]) + U[:, None, :] * (1 - PM[None])
    L = np.where((ans == 1)[:, None, :], pres, np.where((ans == -1)[:, None, :], absn, 1.0))
    L = np.where(M[:, None, :], L, 1.0)
    post = np.prod(L, axis=2)
    return post / post.sum(axis=1, keepdims=True)


def simulate_interview(n_per, p_obs, p_fp, conf, rng):
    y = np.repeat(np.arange(NP), n_per)
    n = len(y)
    T = L0[y] > 0                                           # symptoms the true disease has
    r = rng.random((n, NG))
    present = np.where(T, r < p_obs, r < p_fp)
    cm, am = CONF_MODELS[conf], ABS_MODELS[conf]
    u_true = rng.choice(cm["true_levels"], size=(n, NG), p=cm["true_p"])
    u_false = rng.choice(cm["false_levels"], size=(n, NG), p=cm["false_p"])
    u_abs_ok = rng.choice(am["correct"][0], size=(n, NG), p=am["correct"][1])
    u_abs_bad = rng.choice(am["wrong"][0], size=(n, NG), p=am["wrong"][1])
    U = np.where(present, np.where(T, u_true, u_false), np.where(T, u_abs_bad, u_abs_ok))
    ans = np.where(present, 1, -1)
    return ans, U, y


def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log2(p)).sum(axis=1)


def adaptive_order(ans, U):
    """Greedy information-gain question order (uniform prior; hard answers for selection, soft update)."""
    n = len(ans)
    q = np.full((n, NP), 1.0 / NP)
    asked = np.zeros((n, NG), bool)
    order = np.zeros((n, NG), int)
    idx = np.arange(n)
    for t in range(NG):
        Pp = q @ PM                                          # (n, NG) P(present)
        h0 = entropy(q)
        qp = q[:, :, None] * PM[None] / np.clip(Pp[:, None, :], 1e-12, None)
        qa = q[:, :, None] * (1 - PM[None]) / np.clip(1 - Pp[:, None, :], 1e-12, None)
        hp = -(np.clip(qp, 1e-12, 1) * np.log2(np.clip(qp, 1e-12, 1))).sum(axis=1)
        ha = -(np.clip(qa, 1e-12, 1) * np.log2(np.clip(qa, 1e-12, 1))).sum(axis=1)
        eig = h0[:, None] - (Pp * hp + (1 - Pp) * ha)
        eig[asked] = -np.inf
        g = eig.argmax(axis=1)
        order[:, t] = g
        asked[idx, g] = True
        u, a = U[idx, g], ans[idx, g]
        pj = PM[:, g].T                                      # (n, NP)
        lik = np.where((a == 1)[:, None], (1 - u)[:, None] + u[:, None] * pj,
                       (1 - u)[:, None] + u[:, None] * (1 - pj))
        q = q * lik
        q = q / q.sum(axis=1, keepdims=True)
    return order


def mask_from_order(order, k):
    n = len(order)
    M = np.zeros((n, NG), bool)
    if k > 0:
        M[np.arange(n)[:, None], order[:, :k]] = True
    return M


def curve(order, ans, U, y, kappa):
    top1, top3 = [], []
    S_all, c1_all = [], []
    for k in range(NG + 1):
        S = signed_scores(ans, U, mask_from_order(order, k), kappa)
        st = rank_stats(S, y)
        top1.append(float(st["top1"].mean())); top3.append(float(st["top3"].mean()))
        S_all.append(S); c1_all.append(st["top1"])
    return top1, top3, S_all, np.stack(c1_all, axis=1)


def stop_stats(S_all, c1, T_CF=0.8, MARGIN=0.2):
    n = c1.shape[0]
    stop = np.full(n, NG)
    done = np.zeros(n, bool)
    for k in range(1, NG + 1):
        S = np.sort(S_all[k], axis=1)
        ok = (S[:, -1] >= T_CF) & (S[:, -1] - S[:, -2] >= MARGIN) & ~done
        stop[ok] = k
        done |= ok
    acc = c1[np.arange(n), stop]
    return float(stop.mean()), float(acc.mean()), float(done.mean())


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "eps_fp": EPS_FP}

    # ---------------- properties of the signed combination (random tests) -------------------
    prng = np.random.default_rng(SEED + 1)
    ok_bound = ok_pos = ok_mono = 0
    T = 3000
    for _ in range(T):
        ans = np.where(prng.random((1, NG)) < 0.4, 1, -1)
        U = prng.choice(LV5, size=(1, NG)).astype(float)
        M = np.ones((1, NG), bool)
        s = signed_scores(ans, U, M, 0.5)
        ok_bound += bool((s >= -1 - 1e-12).all() and (s <= 1 + 1e-12).all())
        ok_pos += bool(np.allclose(signed_scores(np.ones_like(ans), U, M, 0.5), score_proposed(U)))
        # raising one present confidence never lowers any CF; raising one absent confidence never raises any CF
        j = prng.integers(NG)
        U2 = U.copy(); U2[0, j] = min(1.0, U2[0, j] + 0.2)
        s2 = signed_scores(ans, U2, M, 0.5)
        ok_mono += bool((s2 >= s - 1e-12).all() if ans[0, j] == 1 else (s2 <= s + 1e-12).all())
    res["props"] = dict(n=T, bounded=ok_bound, reduces_to_positive=ok_pos, monotone=ok_mono)

    # ---------------- C. comparison with NB and DS ------------------------------------------
    conds = [("calibrated", 0.8, 0.05), ("uninformative", 0.8, 0.05), ("calibrated", 0.8, 0.20), ("calibrated", 0.6, 0.10)]
    comp = {}
    for conf, po, pf in conds:
        U, y = simulate(2000, po, pf, conf, rng)
        row = {}
        for name, S in (("Proposed FC+CF", score_proposed(U)), ("Naive Bayes", score_nb(U)),
                        ("Dempster-Shafer", score_ds(U))):
            st = rank_stats(S, y)
            row[name] = {k: mean_ci(v) for k, v in st.items()}
        for eps in (0.01, 0.20):
            st = rank_stats(score_nb(U, eps), y)
            row[f"Naive Bayes (eps={eps})"] = {k: mean_ci(v) for k, v in st.items()}
        comp[f"{conf}|{po}|{pf}"] = row
    res["C_comparison"] = comp

    # ---------------- B/A. interview experiments ---------------------------------------------
    N_PER = 1500
    interview = {}
    for conf, po, pf in [("calibrated", 0.8, 0.05), ("uninformative", 0.8, 0.05), ("calibrated", 0.6, 0.10)]:
        ans, U, y = simulate_interview(N_PER, po, pf, conf, rng)
        n = len(y)
        full = np.ones((n, NG), bool)
        out = {"n": n}
        # B: kappa sweep at the full checklist
        sw = {}
        for kap in (0.0, 0.25, 0.5, 0.75, 1.0):
            st = rank_stats(signed_scores(ans, U, full, kap), y)
            sw[str(kap)] = dict(top1=mean_ci(st["top1"]), top3=mean_ci(st["top3"]))
        st = rank_stats(score_nb_interview(ans, U, full), y)
        sw["NB"] = dict(top1=mean_ci(st["top1"]), top3=mean_ci(st["top3"]))
        out["kappa_sweep"] = sw
        # A: question orders
        orders = {
            "fixed": np.tile(np.arange(NG), (n, 1)),
            "random": np.stack([rng.permutation(NG) for _ in range(n)]),
            "adaptive": adaptive_order(ans, U),
        }
        curves, stops = {}, {}
        for name, od in orders.items():
            t1, t3, S_all, c1 = curve(od, ans, U, y, 0.5)
            curves[name] = dict(top1=t1, top3=t3)
            stops[name] = dict(zip(("mean_questions", "accuracy", "stopped_share"), stop_stats(S_all, c1)))
            if name == "adaptive":
                t1p, t3p, S_p, c1p = curve(od, ans, U, y, 0.0)
                curves["adaptive_positive_only"] = dict(top1=t1p, top3=t3p)
                stops["adaptive_positive_only"] = dict(zip(("mean_questions", "accuracy", "stopped_share"),
                                                          stop_stats(S_p, c1p)))
        out["curves"] = curves
        out["stopping"] = stops
        # first questions asked by the adaptive policy (most frequent symptom per position)
        od = orders["adaptive"]
        out["adaptive_first"] = [[SYMPTOMS[int(np.bincount(od[:, t], minlength=NG).argmax())],
                                  float(np.bincount(od[:, t], minlength=NG).max() / n)] for t in range(4)]
        interview[f"{conf}|{po}|{pf}"] = out
    res["AB_interview"] = interview

    with open("results_ext.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    make_fig(res)
    print("done")


def make_fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.6})
    sty = {"adaptive": dict(color="black", marker="o", ls="-", label="Adaptive (information gain)"),
           "fixed": dict(color="0.4", marker="s", ls="--", label="Fixed order (G1, G2, \u2026)"),
           "random": dict(color="0.65", marker="^", ls=":", label="Random order"),
           "adaptive_positive_only": dict(color="black", marker="x", ls="-.", label="Adaptive, positive evidence only")}
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.4), sharey=True)
    for ax, key, title in zip(axes, ["calibrated|0.8|0.05", "uninformative|0.8|0.05"],
                              ["(a) calibrated confidence", "(b) uninformative confidence"]):
        for name, s in sty.items():
            ax.plot(range(NG + 1), [v * 100 for v in res["AB_interview"][key]["curves"][name]["top1"]],
                    lw=1.0, ms=3, **s)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Questions asked"); ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("fig_adaptive.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
