import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time

def update_tau(tau, sigma2):
    """Posterior variance after one observation (Gaussian conjugate)."""
    return (tau * sigma2) / (tau + sigma2)

def build_transition_kernel(ms, var_mprime, N, dx):
    """
    Build the transition kernel matrix P where
    P[i, j] ≈ P(m' = ms[j] | current m = ms[i]) * dx
    (so that P.dot(V) approximates E[V(m') | m]).
    """
    std = math.sqrt(max(var_mprime, 1e-14))
    loc = ms.reshape((N, 1))
    x = ms.reshape((1, N))
    pdf = norm.pdf(x, loc, std)
    P = pdf * dx
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    return P

def solve_value_iteration_for_tau(c, tau, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=None, verbose=False):
    """
    Solve the infinite-horizon Bellman for fixed restart-cost c and fixed current tau.
    Returns V(m) on grid ms.
    If P (transition kernel) is provided, uses it; otherwise builds kernel from tau.
    """
    tau_prime = update_tau(tau, sigma2)
    var_mprime = max(tau - tau_prime, 0.0)

    if P is None:
        P = build_transition_kernel(ms, var_mprime, N, dx)

    V = np.maximum(ms, 0.0)
    for it in range(1, max_inner + 1):
        EV = P.dot(V)
        cont = -c + beta * EV
        V_new = np.maximum(np.maximum(0.0, ms), cont)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < inner_tol:
            if verbose:
                print(f"  value-iter converged after {it} iters (diff {diff:.2e}) for tau={tau:.4f}")
            return V, P, it
    if verbose:
        print(f"  WARN: value-iter reached max_inner={max_inner} (last diff {diff:.2e}) for tau={tau:.4f}")
    return V, P, max_inner

def extract_thresholds_from_V_and_cont(V, cont, ms, N):
    """
    Given V and continuation value cont (vectors on grid), extract thresholds:
    a = max m where action == reject (0)
    b = min m where action == buy (m)
    If no reject/buy region exists returns None for that threshold.
    """
    dx = (ms[-1] - ms[0]) / (N - 1)
    actions = np.empty(N, dtype=object)
    for i, m in enumerate(ms):
        vals = (0.0, m, cont[i])
        best = max(vals)
        if best == 0.0:
            actions[i] = "reject"
        elif best == m:
            actions[i] = "buy"
        else:
            actions[i] = "continue"

    rej_idx = np.where(actions == "reject")[0]
    buy_idx = np.where(actions == "buy")[0]
    a = (ms[rej_idx.max()] if len(rej_idx) > 0 else None)
    b = (ms[buy_idx.min()] if len(buy_idx) > 0 else None)
    return a, b, actions, dx

def value_iteration_with_c(tau0=1.0, mu0=-1.0, sigma2=1.0, beta=0.95,
                           max_outer=100, damping=0.8, outer_tol=1e-7,
                           max_inner=5000, inner_tol=1e-9,
                           T=40, plots=False,
                           m_min=-10.0, m_max=10.0, N=1001):
    """
    Main function to perform value iteration with fixed-point search for c,
    and compute thresholds a_t, b_t for t = 1..T.
    
    Returns: taus, a_list, b_list, c (for use by other functions)
    """
    # Grid setup
    dx = (m_max - m_min) / (N - 1)
    ms = np.linspace(m_min, m_max, N)
    
    # -----------------------------
    # 0) Compute tau1 and distribution of m1 (posterior mean after first signal)
    # -----------------------------
    tau1 = update_tau(tau0, sigma2)                # posterior variance after 1st observation
    var_m1 = max(tau0 - tau1, 0.0)                # Var(m1) = tau0 - tau1
    sd_m1 = math.sqrt(var_m1)

    # -----------------------------
    # 1) Fixed-point iteration for c using tau1 (decisions begin after first signal)
    # -----------------------------
    start_time = time.time()

    # Pre-build transition kernel for tau1 (this is the kernel used in stationary DP)
    tau2 = update_tau(tau1, sigma2)
    var_mprime1 = max(tau1 - tau2, 0.0)
    P1 = build_transition_kernel(ms, var_mprime1, N, dx)

    c = 0.05    # start with 0 is sensible when shifting to t=1
    c_history = [c]
    V_current = None
    outer_iter = 0
    for outer_iter in range(1, max_outer + 1):
        # Solve DP for current candidate c using tau1 (decisions after first signal)
        V_current, _, inner_iters = solve_value_iteration_for_tau(c, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=False)

        # Instead of V(mu0) we need E[V(m1)] where m1 ~ N(mu0, var_m1)
        prior_m1_pdf = norm.pdf(ms, loc=mu0, scale=sd_m1)
        weights_m1 = prior_m1_pdf * dx
        V_mu1_mean = np.sum(V_current * weights_m1)

        c_new = beta * V_mu1_mean
        # damped update
        c = damping * c + (1.0 - damping) * c_new
        c_history.append(c)
        if abs(c_history[-1] - c_history[-2]) < outer_tol:
            break

    # final accurate solve for tau1
    V_stationary, P1, inner_iters_final = solve_value_iteration_for_tau(c, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=True)

    # compute EV and cont for tau1 to extract thresholds (these are thresholds used at t=1)
    EV1 = P1.dot(V_stationary)
    cont1 = -c + beta * EV1
    a1, b1, actions1, _ = extract_thresholds_from_V_and_cont(V_stationary, cont1, ms, N)

    elapsed = time.time() - start_time
    print(f"Fixed-point finished in {outer_iter} outer iters; final c = {c:.6f}; elapsed {elapsed:.2f}s")
    print(f"Thresholds after first signal (t=1): a1 = {a1}, b1 = {b1}")

    # -----------------------------
    # 2) Compute taus for t=1..T and compute a_t, b_t for each tau_t
    #    (we reuse the same c; for each tau we rebuild P and solve Bellman with that tau)
    # -----------------------------
    taus = [tau1]
    for t in range(1, T):
        taus.append(update_tau(taus[-1], sigma2))

    a_list = []
    b_list = []
    for t_idx, tau_t in enumerate(taus):   # t_idx=0 corresponds to t=1
        tau_prime_t = update_tau(tau_t, sigma2)
        var_mprime_t = max(tau_t - tau_prime_t, 0.0)
        P_t = build_transition_kernel(ms, var_mprime_t, N, dx)
        V_t, _, _ = solve_value_iteration_for_tau(c, tau_t, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P_t, verbose=False)
        EV_t = P_t.dot(V_t)
        cont_t = -c + beta * EV_t
        a_t, b_t, _, _ = extract_thresholds_from_V_and_cont(V_t, cont_t, ms, N)
        a_list.append(a_t)
        b_list.append(b_t)

    # Determine the first t (indexing corresponds to t=1 + index) where a_t == b_t
    t_no_continue = None
    for idx, (aa, bb) in enumerate(zip(a_list, b_list)):
        if (aa is None) or (bb is None):
            t_no_continue = idx + 1   # add 1 so this reports the actual t (since idx 0 -> t=1)
            break
        if abs(aa - bb) <= dx * 1.001:
            t_no_continue = idx + 1
            break

    # -----------------------------
    # 3) Outputs & Plots
    # -----------------------------
    print("\nSummary:")
    print(f"Parameters: beta={beta}, sigma2={sigma2}, mu0={mu0}, tau0={tau0}")
    print(f"Stationary restart cost c = {c:.6f}")
    print(f"Thresholds at t=1: a1={a1}, b1={b1}")
    if t_no_continue is None:
        print("No collapse (a_t != b_t) found for t in 1..T.")
    else:
        print(f"First t with collapsed thresholds (no 'continue' region) = {t_no_continue}")

    # compute population expected utility under the prior AFTER first signal, i.e. E[V(m1)]
    prior_m1_pdf = norm.pdf(ms, loc=mu0, scale=sd_m1)   # density of m1 at each ms
    weights_m1 = prior_m1_pdf * dx
    EU_pop = np.sum(V_stationary * weights_m1)

    # compute one-period baseline (decision at t=1, beta->0): E[max(0,m1)]
    EU_oneperiod = sd_m1 * norm.pdf(-mu0/sd_m1) + mu0 * norm.cdf(mu0/sd_m1)

    print(f"V evaluated at prior-mean (for reference) = {np.interp(mu0, ms, V_stationary):.6f}")
    print(f"Population expected utility after first signal EU_pop = {EU_pop:.6f}")
    print(f"Baseline one-period EU (decision at t=1, beta->0) EU_oneperiod = {EU_oneperiod:.6f}")

    # Print a table of taus and thresholds, label t from 1..T
    t_vals = np.arange(1, 1 + len(taus))
    df = pd.DataFrame({
        "t": t_vals,
        "tau_t": taus,
        "a_t": a_list,
        "b_t": b_list
    })
    print("\nThreshold table (t, tau_t, a_t, b_t):")
    print(df.to_string(index=False))

    # Optional plots
    if plots:
        # Plot V(m) with thresholds for tau1
        plt.figure(figsize=(9,5))
        plt.plot(ms, V_stationary, label='Temporal-aware return')
        plt.plot(ms, np.maximum(ms, 0.0), linestyle='--', label='Naive return', color='black')
        if a1 is not None:
            plt.axvline(a1, color='red', linestyle=':', label=f'Shelve/Continue={a1:.3f}')
        if b1 is not None:
            plt.axvline(b1, color='green', linestyle=':', label=f'Continue/Ship={b1:.3f}')
        plt.xlim(-1.5, 1.5)  # Restrict x-axis range
        plt.xlabel('Posterior mean')
        plt.ylabel('Expected return')
        plt.title('Expected returns vs posterior mean (after one test/signal)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot a_t and b_t vs t (t=1..)
        plt.figure(figsize=(8,4))
        ts = t_vals
        a_vals = np.array([np.nan if x is None else x for x in a_list])
        b_vals = np.array([np.nan if x is None else x for x in b_list])
        plt.plot(ts, a_vals, marker='o', markersize=2, label='Shelve/Continue threshold')
        plt.plot(ts, b_vals, marker='o',  markersize=2, color='black', label='Continue/Ship threshold')
        plt.xlabel('Number of tests/signals accumulated (per unit)')
        plt.ylabel('Posterior mean threshold')
        plt.title('Decision Thresholds')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 10)  # Restrict x-axis range
        plt.xticks(np.arange(min(ts), 11, 1))
        plt.show()

        # Plot c convergence
        plt.figure(figsize=(6,3))
        plt.plot(range(len(c_history)), c_history, marker='o')
        plt.xlabel('outer iteration')
        plt.ylabel('c')
        plt.title('Convergence of restart cost c')
        plt.grid(True)
        plt.show()
    
    return taus, a_list, b_list, c, ms, dx, N

def compute_action_probabilities(taus, a_list, b_list, mu0=-1.0, plots=False):
    # ----------------------------------------------------------
    # Compute probabilities (for t = 1..)
    # ----------------------------------------------------------
    rej_probs = []
    buy_probs = []
    cont_probs = []

    for idx in range(len(taus)):
        a_t = a_list[idx]
        b_t = b_list[idx]
        tau_t = taus[idx]
        sd = math.sqrt(tau_t)

        rej = norm.cdf((a_t - mu0)/sd)
        buy = 1 - norm.cdf((b_t - mu0)/sd)
        cont = 1 - rej - buy

        rej_probs.append(rej)
        buy_probs.append(buy)
        cont_probs.append(cont)

    # Put into a dataframe for convenience, t labelled from 1..
    t_vals = np.arange(1, 1 + len(taus))
    df_probs = pd.DataFrame({
        "t": t_vals,
        "Reject": rej_probs,
        "Continue": cont_probs,
        "Buy": buy_probs
    })

    # Optional plots
    if plots:
        # Survival probability that decision process reaches time t (S[0] for t=1)
        S = np.ones_like(df_probs["Reject"])
        for i in range(1, len(S)):
            S[i] = S[i-1] * df_probs["Continue"][i-1]

        # Per-time occurrence probabilities (hazard * survival)
        rej_occ = df_probs["Reject"] * S
        buy_occ = df_probs["Buy"] * S

        # Cumulative probabilities up to each t
        rej_cum = np.cumsum(rej_occ)
        buy_cum = np.cumsum(buy_occ)

        #print("\nAction probabilities (for decisions at t=1..):")
        #print(df_probs)

        plt.figure(figsize=(10,6))
        plt.plot(df_probs["t"], df_probs["Reject"], label="Shelve", marker='o', markersize=2, color='red')
        plt.plot(df_probs["t"], df_probs["Continue"], label="Continue", marker='o', markersize=2, color='blue')
        plt.plot(df_probs["t"], df_probs["Buy"], label="Ship", marker='o', markersize=2, color='green')
        plt.plot(df_probs["t"], rej_cum, label='Shelve (cumulative)', color='darkred', linestyle='--')
        plt.plot(df_probs["t"], buy_cum, label='Ship (cumulative)', color='darkgreen', linestyle='--')

        plt.xlabel("Number of tests/signals accumulated (per unit)")
        plt.ylabel("Probability")
        plt.title("Action probabilities (per unit, per signal)")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 10)  # Restrict x-axis range
        plt.xticks(np.arange(min(df_probs["t"]), 11, 1))
        plt.show()

    return df_probs


# ============================================================
#  NEW: helper that runs the full DP for a given i
#       using sigma_i^2 = sigma2 * i / N_total
# ============================================================

def run_dp_for_i(i,
                 N_total,
                 mu0,
                 tau0,
                 beta,
                 sigma2_base,
                 ms,
                 dx,
                 N,
                 max_outer=80,
                 outer_tol=1e-7,
                 max_inner=5000,
                 inner_tol=1e-9,
                 damping=0.8,
                 T=40,
                 z=0.0):

    # ---------------------------
    # 1. Compute sigma_i^2 and updated tau transitions
    # ---------------------------
    sigma2_i = sigma2_base * (i / N_total)

    def update_tau_i(tau):
        return (tau * sigma2_i) / (tau + sigma2_i)

    # ---------------------------
    # 2. Build transition kernels (local version using N from closure)
    # ---------------------------
    def build_transition_kernel_local(ms, var_mprime):
        std = math.sqrt(max(var_mprime, 1e-14))
        loc = ms.reshape((N, 1))
        x = ms.reshape((1, N))
        pdf = norm.pdf(x, loc, std)
        P = pdf * dx
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P /= row_sums
        return P

    # ---------------------------
    # 3. Value iteration for fixed tau (local version)
    # ---------------------------
    def solve_value_iteration_local(c, tau, P=None):
        tau_prime = update_tau_i(tau)
        var_mprime = max(tau - tau_prime, 0.0)

        if P is None:
            P = build_transition_kernel_local(ms, var_mprime)

        V = np.maximum(ms, 0.0)
        for it in range(1, max_inner + 1):
            EV = P.dot(V)
            cont = -c + beta * EV
            V_new = np.maximum(np.maximum(0.0, ms), cont)
            if np.max(np.abs(V_new - V)) < inner_tol:
                return V_new, P
            V = V_new
        return V, P  # return even if not converged

    # ---------------------------
    # 4. Compute tau1(i) & distribution of m1
    # ---------------------------
    tau1_i = update_tau_i(tau0)
    var_m1_i = max(tau0 - tau1_i, 0.0)
    sd_m1_i = math.sqrt(var_m1_i)

    # Precompute transition kernel for DP
    tau2_i = update_tau_i(tau1_i)
    var_mprime1_i = max(tau1_i - tau2_i, 0.0)
    P1_i = build_transition_kernel_local(ms, var_mprime1_i)

    # ---------------------------
    # 5. Outer fixed-point iteration to find c(i)
    # ---------------------------
    c = 0.05
    for _ in range(max_outer):
        V_i, _ = solve_value_iteration_local(c, tau1_i, P=P1_i)

        weights_m1 = norm.pdf(ms, loc=mu0, scale=sd_m1_i) * dx
        EV_m1 = np.sum(V_i * weights_m1)

        c_new = beta * EV_m1 - z
        c_damped = damping * c + (1 - damping) * c_new

        if abs(c_damped - c) < outer_tol:
            c = c_damped
            break
        c = c_damped

    # ---------------------------
    # 6. Final DP solve for tau1(i)
    # ---------------------------
    V_i, P1_i = solve_value_iteration_local(c, tau1_i, P=P1_i)

    # ---------------------------
    # 7. Population expected utility for this i
    # ---------------------------
    weights_m1 = norm.pdf(ms, loc=mu0, scale=sd_m1_i) * dx
    EU_pop_i = np.sum(V_i * weights_m1)

    return EU_pop_i, c, tau1_i, V_i


def optimize_over_i(
    N_total,
    mu0,
    tau0,
    beta,
    sigma2,
    ms,
    dx,
    N,
    max_outer=80,
    outer_tol=1e-7,
    max_inner=5000,
    inner_tol=1e-9,
    damping=0.8,
    T=40,
    plots=True):

    i_values = range(1, 200)
    results = []   # (i, EU_i, L_i, F_i, EV_i)

    for i in i_values:

        # ================================
        # 1. Run dynamic programming
        # ================================
        EU_i, c_i, tau1_i, V_i = run_dp_for_i(
            i=i,
            N_total=N_total,
            mu0=mu0,
            tau0=tau0,
            beta=beta,
            sigma2_base=sigma2,
            ms=ms,
            dx=dx,
            N=N,
            max_outer=max_outer,
            outer_tol=outer_tol,
            max_inner=max_inner,
            inner_tol=inner_tol,
            damping=damping,
            T=T,
        )

        # =======================================
        # 2. Compute action probabilities for THIS i
        # =======================================
        df_probs = compute_action_probabilities(
            taus=tau1_i,      # or taus used inside DP
            a_list=c_i,       # depends on your original structure
            b_list=None,      # if applicable
            mu0=mu0,
            plots=False
        )

        pis = df_probs["Continue"].values
        Tmax = len(pis)

        # compute φ_t
        phis = np.ones(Tmax)
        for t in range(1, Tmax):
            phis[t] = phis[t-1] * pis[t-1]

        # expected lifetime for THIS i
        L_i = np.sum(phis)

        # steady-state inflow
        F_i = i / L_i

        # steady-state expected value
        EV_i = F_i * EU_i

        results.append((i, EU_i, L_i, F_i, EV_i))

    # ======================================
    # Choose i maximizing EV_i
    # ======================================
    best = max(results, key=lambda x: x[4])
    best_i, best_EU, best_L, best_F, best_EV = best

    print(f"Best i = {best_i}")
    print(f"EU(i) = {best_EU:.6f}")
    print(f"L(i) = {best_L:.6f}")
    print(f"F(i) = {best_F:.6f}")
    print(f"EV(i) = {best_EV:.6f}")

    # ================================
    # Plot EV(i)
    # ================================
    if plots:
        plt.figure(figsize=(6,4))
        plt.plot([r[0] for r in results], [r[4] for r in results])
        plt.axvline(best_i, color='red', ls='--', label=f'i*={best_i}')
        plt.xlabel("Number of ideas tested per period (i)")
        plt.ylabel("Steady-state expected value EV(i)")
        plt.title("Implied steady-state expected value vs i")
        plt.grid(True)
        plt.legend()
        plt.show()

    return best_i, best_EU, best_L, best_F, best_EV



# ----------------------------------------------------------
# Compute the expected lifetime L and the steady-state inflow F
# ----------------------------------------------------------

def compute_steady_state_values(
    taus, a_list, b_list,
    N_total, mu0, tau0, beta, sigma2, ms,
    dx, N,
    max_outer=80, outer_tol=1e-7,
    max_inner=5000, inner_tol=1e-9,
    damping=0.8, T=40, z=0.0):

    # Compute action probabilities using the computed thresholds
    df_probs = compute_action_probabilities(
        taus=taus,
        a_list=a_list,
        b_list=b_list,
        mu0=mu0,
        plots=False
    )
    
    # Compute expected lifetime L from continuation probabilities
    pis = df_probs["Continue"].values
    Tmax = len(pis)
    
    # compute φ_t (survival probability)
    phis = np.ones(Tmax)
    for t in range(1, Tmax):
        phis[t] = phis[t-1] * pis[t-1]
    
    # expected lifetime 
    L = np.sum(phis)
    
    # Get the population expected utility after first signal
    tau1 = taus[0]  # posterior variance after 1st observation
    var_m1 = max(tau0 - tau1, 0.0)
    sd_m1 = math.sqrt(var_m1)
    
    # Pre-build transition kernel for tau1
    tau2 = update_tau(tau1, sigma2)
    var_mprime1 = max(tau1 - tau2, 0.0)
    P1 = build_transition_kernel(ms, var_mprime1, N, dx)
    
    # Solve for the value function at steady state (reusing tau1)
    # Find c through fixed point
    c = 0.05
    for _ in range(max_outer):
        V_current, _, _ = solve_value_iteration_for_tau(c, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=False)
        prior_m1_pdf = norm.pdf(ms, loc=mu0, scale=sd_m1)
        weights_m1 = prior_m1_pdf * dx
        V_mu1_mean = np.sum(V_current * weights_m1)
        c_new = beta * V_mu1_mean - z
        c = damping * c + (1.0 - damping) * c_new
        if abs(c - c_new) < outer_tol:
            break
    
    # Final solve
    V_stationary, P1, _ = solve_value_iteration_for_tau(c, tau1, ms, sigma2, beta, max_inner, inner_tol, N, dx, P=P1, verbose=False)
    
    # Population expected utility
    prior_m1_pdf = norm.pdf(ms, loc=mu0, scale=sd_m1)
    weights_m1 = prior_m1_pdf * dx
    EU_pop = np.sum(V_stationary * weights_m1)
    
    # =========================================================================
    # UPDATED: Search over values of j (number of ideas to test concurrently)
    # to find the optimal allocation. We sample key values to balance accuracy
    # and computational efficiency.
    # =========================================================================
    print("\n--- Searching for optimal j (number of ideas to test concurrently) ---")
    
    # Search over a strategic subset: key values around N_total
    j_max = 10

    results = []
    
    for j_candidate in range(1, j_max + 1):
        print(f"  Computing for j={j_candidate}...")
        
        # Run full DP for this j
        EU_j, c_j, tau1_j, V_j = run_dp_for_i(
            i=j_candidate,
            N_total=N_total,
            mu0=mu0,
            tau0=tau0,
            beta=beta,
            sigma2_base=sigma2,
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
        
        # Compute lifetime for this j
        sigma2_j = sigma2 * (j_candidate / N_total)
        
        def update_tau_j(tau):
            return (tau * sigma2_j) / (tau + sigma2_j)
        
        # Build taus for this j
        taus_j = [tau1_j]
        for _ in range(1, T):
            taus_j.append(update_tau_j(taus_j[-1]))
        
        # Extract thresholds for this j
        a_list_j = []
        b_list_j = []
        
        for tau_t in taus_j:
            tau_prime_t = update_tau_j(tau_t)
            var_mprime_t = max(tau_t - tau_prime_t, 0.0)
            P_t = build_transition_kernel(ms, var_mprime_t, N, dx)
            V_t, _, _ = solve_value_iteration_for_tau(c_j, tau_t, ms, sigma2_j, beta, max_inner, inner_tol, N, dx, P=P_t, verbose=False)
            EV_t = P_t.dot(V_t)
            cont_t = -c_j + beta * EV_t
            a_t, b_t, _, _ = extract_thresholds_from_V_and_cont(V_t, cont_t, ms, N)
            a_list_j.append(a_t)
            b_list_j.append(b_t)
        
        # Compute action probabilities and lifetime for this j
        df_probs_j = compute_action_probabilities(
            taus=taus_j,
            a_list=a_list_j,
            b_list=b_list_j,
            mu0=mu0,
            plots=False
        )
        
        pis_j = df_probs_j["Continue"].values
        Tmax_j = len(pis_j)
        
        # Compute lifetime
        phis_j = np.ones(Tmax_j)
        for t in range(1, Tmax_j):
            phis_j[t] = phis_j[t-1] * pis_j[t-1]
        
        L_j = np.sum(phis_j)
        
        # Steady-state inflow
        F_j = j_candidate / L_j
        
        # Steady-state expected value
        EV_j = F_j * EU_j
        
        results.append((j_candidate, EU_j, L_j, F_j, EV_j))
    
    # Find the best j
    best = max(results, key=lambda x: x[4])
    best_j, best_EU, best_L, best_F, best_EV = best
    
    print(f"\nOptimal j = {best_j}")
    print(f"Expected utility per idea EU(j*) = {best_EU:.6f}")
    print(f"Expected lifetime L(j*) = {best_L:.4f}")
    print(f"Steady-state inflow F(j*) = {best_F:.4f}")
    print(f"Steady-state expected value EV(j*) = {best_EV:.6f}")
    
    # Return best values and also the full results for plotting
    return best_L, best_F, best_EV, best_j, best_EU, results

