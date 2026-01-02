"""
Cost Analysis: Variant with cost of idea generation
Compares standard (sequential Bayesian) case with version including idea generation cost
"""

import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

from Code.Temporal_aware import (
    value_iteration_with_c,
    compute_action_probabilities,
    compute_steady_state_values,
    build_transition_kernel,
    update_tau,
    solve_value_iteration_for_tau,
    extract_thresholds_from_V_and_cont,
)

# Standard parameters
mu0 = -1.0
tau0 = 2.0
sigma2 = 8.0
beta = 0.99
N_total = 1

# Grid settings
m_min, m_max = -2.0, 2.0
N = 1001
dx = (m_max - m_min) / (N - 1)
ms = np.linspace(m_min, m_max, N)

# Solver settings
max_outer = 80
damping = 0.9
outer_tol = 1e-7
max_inner = 5000
inner_tol = 1e-9
T = 20

# Cost parameter
z = 0.01  # Cost of generating a new idea

print("="*60)
print("COST ANALYSIS: Idea Generation Cost")
print("="*60)
print(f"\nParameters: mu0={mu0}, tau0={tau0}, sigma2={sigma2}, beta={beta}, N={N_total}")
print(f"Idea generation cost: z = {z}")

# ============================================================
# 1. Run standard case (no cost)
# ============================================================
print("\n" + "="*60)
print("1. STANDARD CASE (No Idea Generation Cost, z=0)")
print("="*60 + "\n")

taus_std, a_list_std, b_list_std, c_std, ms_std, dx_std, N_std = value_iteration_with_c(
    tau0=tau0,
    mu0=mu0,
    sigma2=sigma2,
    beta=beta,
    max_outer=max_outer,
    damping=damping,
    outer_tol=outer_tol,
    max_inner=max_inner,
    inner_tol=inner_tol,
    T=T,
    plots=False,
    m_min=m_min,
    m_max=m_max,
    N=N
)

df_probs_std = compute_action_probabilities(
    taus=taus_std,
    a_list=a_list_std,
    b_list=b_list_std,
    mu0=mu0,
    plots=False
)

# ============================================================
# 2. Run case with idea generation cost
# ============================================================
print("\n" + "="*60)
print(f"2. WITH IDEA GENERATION COST (z={z})")
print("="*60 + "\n")

# Modify value iteration to account for opportunity cost

tau1 = update_tau(tau0, sigma2)
var_m1 = max(tau0 - tau1, 0.0)
sd_m1 = math.sqrt(var_m1)

# Pre-build transition kernel for tau1
tau2 = update_tau(tau1, sigma2)
var_mprime1 = max(tau1 - tau2, 0.0)
P1 = build_transition_kernel(ms, var_mprime1, N, dx)

# Fixed-point iteration with cost
c_cost = 0.05
c_history_cost = [c_cost]
for outer_iter in range(1, max_outer + 1):
    # Solve DP for current candidate c
    V_current, _, inner_iters = solve_value_iteration_for_tau(
        c_cost, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=False
    )

    # E[V(m1)] where m1 ~ N(mu0, var_m1)
    prior_m1_pdf = norm.pdf(ms, loc=mu0, scale=sd_m1)
    weights_m1 = prior_m1_pdf * dx
    V_mu1_mean = np.sum(V_current * weights_m1)

    # Modified fixed-point condition: c' = beta * E[V(m1)] - z
    c_new = beta * V_mu1_mean - z

    # Damped update
    c_cost = damping * c_cost + (1.0 - damping) * c_new
    c_history_cost.append(c_cost)
    if abs(c_history_cost[-1] - c_history_cost[-2]) < outer_tol:
        break

# Final solve with cost
V_cost, P1, inner_iters_final = solve_value_iteration_for_tau(
    c_cost, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=True
)

# Extract thresholds
EV1_cost = P1.dot(V_cost)
cont1_cost = -c_cost + beta * EV1_cost
a1_cost, b1_cost, actions1_cost, _ = extract_thresholds_from_V_and_cont(V_cost, cont1_cost, ms, N)

print(f"Fixed-point converged in {outer_iter} iterations; final c_cost = {c_cost:.6f}")
print(f"Thresholds with cost at t=1: a1 = {a1_cost}, b1 = {b1_cost}")

# Compute taus and thresholds for t=1..T with cost
taus_cost = [tau1]
for t in range(1, T):
    taus_cost.append(update_tau(taus_cost[-1], sigma2))

a_list_cost = []
b_list_cost = []
for t_idx, tau_t in enumerate(taus_cost):
    tau_prime_t = update_tau(tau_t, sigma2)
    var_mprime_t = max(tau_t - tau_prime_t, 0.0)
    P_t = build_transition_kernel(ms, var_mprime_t, N, dx)
    V_t, _, _ = solve_value_iteration_for_tau(c_cost, tau_t, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P_t, verbose=False)
    EV_t = P_t.dot(V_t)
    cont_t = -c_cost + beta * EV_t
    a_t, b_t, _, _ = extract_thresholds_from_V_and_cont(V_t, cont_t, ms, N)
    a_list_cost.append(a_t)
    b_list_cost.append(b_t)

df_probs_cost = compute_action_probabilities(
    taus=taus_cost,
    a_list=a_list_cost,
    b_list=b_list_cost,
    mu0=mu0,
    plots=False
)

# ============================================================
# 3. Compute steady-state values for both cases
# ============================================================
print("\n" + "="*60)
print("3. STEADY-STATE VALUES")
print("="*60 + "\n")

# Standard case
L_std, F_std, EV_std, j_std, EU_std, results_std = compute_steady_state_values(
    taus=taus_std,
    a_list=a_list_std,
    b_list=b_list_std,
    N_total=N_total,
    mu0=mu0,
    tau0=tau0,
    beta=beta,
    sigma2=sigma2,
    ms=ms_std,
    dx=dx_std,
    N=N_std,
    max_outer=max_outer,
    outer_tol=outer_tol,
    max_inner=max_inner,
    inner_tol=inner_tol,
    damping=damping,
    T=T
)

print("\n" + "-"*60)
print("WITH COST:")
print("-"*60)

# Cost case
L_cost, F_cost, EV_cost, j_cost, EU_cost, results_cost = compute_steady_state_values(
    taus=taus_cost,
    a_list=a_list_cost,
    b_list=b_list_cost,
    N_total=N_total,
    mu0=mu0,
    tau0=tau0,
    beta=beta,
    sigma2=sigma2,
    ms=ms,
    dx=dx,
    N=N,
    max_outer=max_outer,
    outer_tol=outer_tol,
    max_inner=max_inner,
    inner_tol=inner_tol,
    damping=damping,
    T=T,
    z=z
)

# ============================================================
# 4. Create comparison plots
# ============================================================
print("\n" + "="*60)
print("4. GENERATING COMPARISON PLOTS")
print("="*60 + "\n")

# Plot 1: Decision Thresholds Comparison
plt.figure(figsize=(10, 6))
t_vals = np.arange(1, len(taus_std) + 1)

# Standard case
a_vals_std = np.array([np.nan if x is None else x for x in a_list_std])
b_vals_std = np.array([np.nan if x is None else x for x in b_list_std])

# Cost case
a_vals_cost = np.array([np.nan if x is None else x for x in a_list_cost])
b_vals_cost = np.array([np.nan if x is None else x for x in b_list_cost])

plt.plot(t_vals, a_vals_std, marker='o', markersize=3, label='Standard (shelve)', 
        linestyle='--', color='blue', alpha=0.7)
plt.plot(t_vals, b_vals_std, marker='s', markersize=3, label='Standard (ship)', 
        linestyle='-', color='blue', alpha=0.7)
plt.plot(t_vals, a_vals_cost, marker='o', markersize=3, label=f'With cost (shelve)', 
        linestyle='--', color='red', alpha=0.7)
plt.plot(t_vals, b_vals_cost, marker='s', markersize=3, label=f'With cost (ship)', 
        linestyle='-', color='red', alpha=0.7)

plt.xlabel('Number of tests accumulated', fontsize=11)
plt.ylabel('Posterior mean threshold', fontsize=11)
plt.title(f'Decision Thresholds: Standard vs With Idea Generation Cost', 
         fontsize=12, fontweight='bold')
plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 1))
plt.tight_layout()
plt.savefig('Thresholds_cost_comparison.png', dpi=150)
print("Saved: Thresholds_cost_comparison.png")
plt.close()

# Plot 2: Optimal j Search Comparison
plt.figure(figsize=(10, 6))

# Standard case
j_vals_std = [r[0] for r in results_std]
EV_vals_std = [r[4] for r in results_std]

# Cost case
j_vals_cost = [r[0] for r in results_cost]
EV_vals_cost = [r[4] for r in results_cost]

plt.plot(j_vals_std, EV_vals_std, label='Standard (no cost)', color='blue', 
        marker='o', markersize=4, alpha=0.7)
plt.plot(j_vals_cost, EV_vals_cost, label=f'With cost', color='red', 
        marker='s', markersize=4, alpha=0.7)

plt.axvline(j_std, color='blue', linestyle='--', alpha=0.5, 
           label=f'Standard optimal j={j_std}')
plt.axvline(j_cost, color='red', linestyle='--', alpha=0.5, 
           label=f'With cost optimal j={j_cost}')

plt.xlabel('Number of ideas tested per period', fontsize=11)
plt.ylabel('Steady-state expected value', fontsize=11)
plt.title(f'Optimal Testing Capacity: Standard vs With Idea Generation Cost', 
         fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)
plt.tight_layout()
plt.savefig('Optimal_j_cost_comparison.png', dpi=150)
print("Saved: Optimal_j_cost_comparison.png")
plt.close()

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"\nStandard (no cost):")
print(f"  Optimal j = {j_std}")
print(f"  Expected lifetime L = {L_std:.4f}")
print(f"  Steady-state EV = {EV_std:.6f}")
print(f"  Thresholds at t=1: a1 = {a_list_std[0]:.3f}, b1 = {b_list_std[0]:.3f}")

print(f"\nWith idea generation cost (z={z}):")
print(f"  Optimal j = {j_cost}")
print(f"  Expected lifetime L = {L_cost:.4f}")
print(f"  Steady-state EV = {EV_cost:.6f}")
print(f"  Thresholds at t=1: a1 = {a_list_cost[0]:.3f}, b1 = {b_list_cost[0]:.3f}")

print(f"\nChanges due to cost:")
print(f"  Δj = {j_cost - j_std} (fewer ideas tested)")
print(f"  ΔL = {L_cost - L_std:.4f} (longer testing per idea)")
print(f"  ΔEV = {EV_cost - EV_std:.6f}")
print(f"  Threshold widening: Δ(b-a) at t=1 = {(b_list_cost[0] - a_list_cost[0]) - (b_list_std[0] - a_list_std[0]):.3f}")

print("\n" + "="*60)
print("COST ANALYSIS COMPLETE")
print("="*60)
