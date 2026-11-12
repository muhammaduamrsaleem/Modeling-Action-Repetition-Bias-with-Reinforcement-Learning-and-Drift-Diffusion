"""
Action Repetition and Decision Bias: a Reinforcement-Learning + Drift-Diffusion Simulation

Interest area: computational psychiatry / decision neuroscience (CaCTüS application project)

Question
--------
Do people (and agents) repeat their previous action more than reward alone justifies,
and how does this repetition tendency show up in BOTH choices and reaction times?

Model
-----
1. Learning: Q-learning on a two-armed bandit, learning rate `alpha`.
2. Choice bias: a repetition bonus `kappa` added to the previously chosen action.
3. Decision: a drift-diffusion process where
       drift = beta * (Q_a - Q_b) + kappa * (+1 if a was repeated option else -1)
   with decision threshold `a_thr` and non-decision time `t0`.
   (Simplified RLDDM in the spirit of hierarchical RL-DDM work; no hierarchy here.)

Pipeline
--------
Part 1  Simulate agents with different (alpha, kappa).
Part 2  Show how kappa changes repetition rate, accuracy, and reaction times.
Part 3  Parameter recovery: fit choice-only model by maximum likelihood and
        check whether alpha, beta, kappa can be recovered from simulated data.
Part 4  Save figures (fig1, fig2, fig3) and print a summary.

Run:  python repetition_bias_rlddm.py
Needs: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RNG = np.random.default_rng(42)


def simulate_agent(alpha, beta, kappa, a_thr=1.0, t0=0.3, n_trials=300,
                   p_reward=(0.7, 0.3), dt=0.005, noise=1.0, rng=RNG):
    """Simulate RL + DDM agent on a two-armed bandit.

    Returns dict with choices, rewards, reaction times, and the correct-arm flags.
    Option 0 is the upper boundary, option 1 the lower boundary.
    """
    q = np.zeros(2)
    prev = None
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    rts = np.zeros(n_trials)

    for t in range(n_trials):
        # repetition term pushes drift toward the previous action
        rep = 0.0
        if prev is not None:
            rep = kappa if prev == 0 else -kappa
        drift = beta * (q[0] - q[1]) + rep

        # Euler simulation of the diffusion to a threshold (+a_thr / -a_thr)
        x, time = 0.0, 0.0
        while abs(x) < a_thr and time < 5.0:
            x += drift * dt + noise * np.sqrt(dt) * rng.standard_normal()
            time += dt
        choice = 0 if x > 0 else 1
        rts[t] = time + t0

        reward = float(rng.random() < p_reward[choice])
        q[choice] += alpha * (reward - q[choice])  # Q-learning update
        choices[t], rewards[t], prev = choice, reward, choice

    return {"choices": choices, "rewards": rewards, "rts": rts,
            "best_arm": int(np.argmax(p_reward))}


def repetition_rate(choices):
    return float(np.mean(choices[1:] == choices[:-1]))


def accuracy(choices, best_arm):
    return float(np.mean(choices == best_arm))


def kappa_sweep(kappas, n_agents=20):
    rows = []
    for k in kappas:
        reps, accs, rt_rep, rt_switch = [], [], [], []
        for _ in range(n_agents):
            d = simulate_agent(alpha=0.3, beta=1.5, kappa=k, a_thr=1.0)
            c, rt = d["choices"], d["rts"]
            reps.append(repetition_rate(c))
            accs.append(accuracy(c, d["best_arm"]))
            same = c[1:] == c[:-1]
            rt_rep.append(rt[1:][same].mean() if same.any() else np.nan)
            rt_switch.append(rt[1:][~same].mean() if (~same).any() else np.nan)
        rows.append((k, np.mean(reps), np.mean(accs),
                     np.nanmean(rt_rep) if not np.all(np.isnan(rt_rep)) else np.nan,
                     np.nanmean(rt_switch) if not np.all(np.isnan(rt_switch)) else np.nan))
    return np.array(rows)

def neg_log_lik(params, choices, rewards):
    alpha, beta, kappa = params
    if not (0.01 <= alpha <= 1 and 0.1 <= beta <= 10):
        return 1e9
    q = np.zeros(2)
    prev, nll = None, 0.0
    for c, r in zip(choices, rewards):
        bonus = np.zeros(2)
        if prev is not None:
            bonus[prev] = kappa
        logits = beta * q + bonus
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        nll -= np.log(max(p[c], 1e-12))
        q[c] += alpha * (r - q[c])
        prev = c
    return nll


def fit_choice_model(choices, rewards):
    best = None
    for start in ([0.3, 2.0, 0.0], [0.6, 4.0, 1.0], [0.1, 1.0, -0.5]):
        res = minimize(neg_log_lik, start, args=(choices, rewards),
                       method="Nelder-Mead", options={"maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def parameter_recovery(n_agents=40):
    """Simulate with the DDM, fit the choice-only softmax model.

    For a DDM with threshold a and unit noise, P(upper) = 1 / (1 + exp(-2*a*drift)),
    so the softmax parameters the fit should recover are
        beta_eff = 2*a*beta,   kappa_eff = 2*a*kappa.
    """
    true, est = [], []
    for _ in range(n_agents):
        a_thr = 1.0
        a = RNG.uniform(0.1, 0.7)
        b = RNG.uniform(0.8, 2.5)
        k = RNG.uniform(-0.3, 0.9)
        d = simulate_agent(a, b, k, a_thr=a_thr, n_trials=400)
        est.append(fit_choice_model(d["choices"], d["rewards"]))
        true.append([a, 2 * a_thr * b, 2 * a_thr * k])
    return np.array(true), np.array(est)


def main():
    print("Part 1-2: sweeping repetition parameter kappa ...")
    kappas = np.linspace(-0.3, 0.9, 7)
    sweep = kappa_sweep(kappas)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(sweep[:, 0], sweep[:, 1], "o-")
    ax[0].set(xlabel="kappa (repetition bonus)", ylabel="P(repeat previous action)",
              title="Repetition rate rises with kappa")
    ax[1].plot(sweep[:, 0], sweep[:, 2], "o-", color="tab:green")
    ax[1].set(xlabel="kappa (repetition bonus)", ylabel="Accuracy (best arm)",
              title="Accuracy as a function of kappa")
    ax[2].plot(sweep[:, 0], sweep[:, 3], "o-", label="repeat trials")
    ax[2].plot(sweep[:, 0], sweep[:, 4], "s--", label="switch trials")
    ax[2].set(xlabel="kappa (repetition bonus)", ylabel="Mean RT (s)",
              title="RTs: repeat vs switch trials")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("fig1_kappa_effects.png", dpi=150)
    plt.close()

    print("Part 3: parameter recovery (this takes a minute) ...")
    true, est = parameter_recovery()
    names = ["alpha (learning rate)", "beta_eff (inverse temp.)",
             "kappa_eff (repetition bonus)"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    recov = []
    for i in range(3):
        r = np.corrcoef(true[:, i], est[:, i])[0, 1]
        recov.append(r)
        ax[i].scatter(true[:, i], est[:, i], alpha=0.7)
        lo, hi = true[:, i].min(), true[:, i].max()
        ax[i].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[i].set(xlabel="true", ylabel="recovered", title=f"{names[i]}\nr = {r:.2f}")
    plt.tight_layout()
    plt.savefig("fig2_parameter_recovery.png", dpi=150)
    plt.close()

    # Example learning curve for one agent with strong repetition
    d = simulate_agent(0.3, 1.5, 0.9, a_thr=1.0)
    run = np.convolve(d["choices"] == d["best_arm"], np.ones(20) / 20, mode="valid")
    plt.figure(figsize=(6, 3.5))
    plt.plot(run)
    plt.xlabel("trial"); plt.ylabel("accuracy (20-trial window)")
    plt.title("Example learning curve (kappa = 1.5)")
    plt.tight_layout()
    plt.savefig("fig3_example_learning_curve.png", dpi=150)
    plt.close()

    print("\n=== SUMMARY ===")
    print("kappa  P(repeat)  accuracy  RT_repeat  RT_switch")
    for row in sweep:
        print(f"{row[0]:5.2f}  {row[1]:9.3f}  {row[2]:8.3f}  {row[3]:9.3f}  {row[4]:9.3f}")
    print("\nParameter recovery (correlation true vs recovered):")
    for n, r in zip(names, recov):
        print(f"  {n}: r = {r:.2f}")
    print("\nFigures saved: fig1_kappa_effects.png, fig2_parameter_recovery.png, "
          "fig3_example_learning_curve.png")


if __name__ == "__main__":
    main()
