"""
Baseline.py

Put gains from Bayesian sequential testing in context:
Compare to a static-style baseline which commits to a fixed testing duration upfront.
Uses simple grid-based solver. Computes thresholds a_t, b_t for tau_t sequence.

"""

import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# -------------------------
# Baseline single-decision optimization over (k,i)
# -------------------------
def baseline_optimal(
    k_max=50,
    i_max=200,
    sigma2=20.0,
    mu0=-2.0,
    tau0=1.0,
    N_total=40,
    beta=0.95,
    plots=False):

    # Expected positive part E[max(X,0)] for X ~ N(mu, sd^2)
    def E_pos_normal(mu, sd):
        if sd <= 0:
            return max(0.0, mu)
        z = mu / sd
        return sd * norm.pdf(-z) + mu * norm.cdf(z)

    results = []
    for k in range(1, k_max + 1):
        for i in range(1, i_max + 1):

            # ===============================
            # Posterior variance calculations
            # ===============================
            sigma2_eff = sigma2 * (i / (k * N_total))
            tau_post = (tau0 * sigma2_eff) / (tau0 + sigma2_eff)

            var_mpost = max(tau0 - tau_post, 0.0)
            sd_mpost = math.sqrt(var_mpost)

            # Expected payoff per unit (one-shot accept/reject)
            EU_unit = E_pos_normal(mu0, sd_mpost)

            # ===============================
            # Steady-state average return
            # ===============================
            L_k = k                     # lifetime = wait k periods
            F_k_i = i / L_k             # inflow rate
            EV_k_i = F_k_i * EU_unit    # steady-state value

            results.append((k, i, EV_k_i, EU_unit, sigma2_eff, tau_post))

    # pick the (k,i) with best steady-state average value
    best = max(results, key=lambda r: r[2])
    best_k, best_i, best_EV, best_EU_unit, best_sigma2_eff, best_tau_post = best

    print("Best (k,i) =", (best_k, best_i))
    print(f"Steady-state EV = {best_EV:.6f}")
    print(f"EU per unit = {best_EU_unit:.6f}")
    print(f"Effective sigma2 = {best_sigma2_eff:.6e}; tau_post = {best_tau_post:.6e}")
    print(f"Lifetime L(k) = {best_k}, Steady-state inflow F = {best_i/best_k:.3f}")

    # small table for review
    topN = 10
    results_sorted = sorted(results, key=lambda r: r[2], reverse=True)
    print("\nTop candidates (k, i, EV, EU_unit):")
    for r in results_sorted[:topN]:
        print(r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}")

    # --------------------------------
    # Visualization (heatmap of EV)
    # --------------------------------
    Kvals = range(1, k_max + 1)
    Ivals = range(1, i_max + 1)
    obj_grid = np.zeros((len(Kvals), len(Ivals)))

    for (k, i, EV, _, _, _) in results:
        obj_grid[k-1, i-1] = EV

    if plots:
        plt.figure(figsize=(10,6))
        plt.imshow(obj_grid, origin='lower', aspect='auto',
                extent=[1, i_max, 1, k_max], cmap='viridis')
        plt.colorbar(label='Steady-state EV')
        plt.scatter([best_i], [best_k], color='red', s=50,
                    label=f'Optimal: (k*={best_k}, i*={best_i})')
        plt.xlabel('Units tested (i)')
        plt.ylabel('Waiting time (k)')
        plt.title('Heatmap of steady-state average returns EV(k,i)')
        plt.legend()
        plt.show()

    return best_k, best_i, best_EV
