import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Import the modules
from Baseline import baseline_optimal
from Temporal_aware import (
    value_iteration_with_c,
    compute_action_probabilities,
    compute_steady_state_values
)

######### PARAMETER SETS #########
# Define three parameter cases:
# μ = -1.0, τ² = 2.0, σ² = 10.0, γ = 0.99, N = 10 (baseline)
# μ = -0.5, τ² = 2.0, σ² = 10.0, γ = 0.99, N = 10 (higher prior mean)
# μ = -1.0, τ² = 1.0, σ² = 10.0, γ = 0.99, N = 10 (lower prior variance)
# Note: N_total represents the testing capacity
param_sets = [
    {'name': r'$\mu=-1.0, \tau^2=2.0$', 'mu0': -1.0, 'tau0': 2.0, 'sigma2': 8.0, 'beta': 0.99, 'N_total': 1},
    {'name': r'$\mu=-0.5, \tau^2=2.0$', 'mu0': -0.5, 'tau0': 2.0, 'sigma2': 8.0, 'beta': 0.99, 'N_total': 1},
    {'name': r'$\mu=-1.0, \tau^2=1.0$', 'mu0': -1.0, 'tau0': 1.0, 'sigma2': 8.0, 'beta': 0.99, 'N_total': 1},
]

k_max = 202                   # search up to k_max periods
i_max = 402                   # search up to i_max units (cap for search)

# Grid / numerical settings
m_min, m_max = -2.0, 2.0   # grid bounds for posterior mean
N = 1001                    # number of grid points. Reduce for speed.

# Fixed-point solver settings
max_outer = 80
damping = 0.8          # damp factor for c updates (0 < damping < 1)
outer_tol = 1e-7

# Value iteration settings
max_inner = 5000
inner_tol = 1e-9

# Number of decision periods (decisions occur at t = 1..T)
T = 20

###################################

if __name__ == "__main__":
    print("="*60)
    print("EVALUATION: Running Baseline and Temporal-Aware Analysis")
    print("="*60)
    
    # Store results for all parameter sets
    all_results = {}
    
    for idx, params in enumerate(param_sets):
        print(f"\n{'='*60}")
        print(f"PARAMETER SET {idx+1}: {params['name']}")
        print(f"{'='*60}")
        
        # Extract parameters
        beta = params['beta']
        sigma2 = params['sigma2']
        mu0 = params['mu0']
        tau0 = params['tau0']
        N_total = params['N_total']
        
        # Compute grid parameters
        dx = (m_max - m_min) / (N - 1)
        ms = np.linspace(m_min, m_max, N)
        
        # ---------------------------------------------------------
        # 1. Run Baseline Optimal
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("1. BASELINE OPTIMAL (Single-Decision Optimization)")
        print("="*60 + "\n")
        
        best_k_base, best_i_base, pp_EU_base = baseline_optimal(
            k_max=k_max, 
            i_max=i_max, 
            sigma2=sigma2, 
            mu0=mu0, 
            tau0=tau0, 
            N_total=N_total, 
            beta=beta, 
            plots=False 
        )
        
        # ---------------------------------------------------------
        # 2. Run Value Iteration with c
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("2. TEMPORAL-AWARE VALUE ITERATION")
        print("="*60 + "\n")
        
        taus, a_list, b_list, c, ms_out, dx_out, N_out = value_iteration_with_c(
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
        
        # ---------------------------------------------------------
        # 3. Compute Action Probabilities
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("3. ACTION PROBABILITIES")
        print("="*60 + "\n")
        
        df_probs = compute_action_probabilities(
            taus=taus,
            a_list=a_list,
            b_list=b_list,
            mu0=mu0,
            plots=False  
        )
        
        # ---------------------------------------------------------
        # 4. Compute Steady-State Values
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("4. STEADY-STATE VALUES")
        print("="*60 + "\n")
        
        L, F, EV_dynamic, best_i_dyn, best_EU, i_search_results = compute_steady_state_values(
            taus=taus,
            a_list=a_list,
            b_list=b_list,
            N_total=N_total,
            mu0=mu0,
            tau0=tau0,
            beta=beta,
            sigma2=sigma2,
            ms=ms_out,
            dx=dx_out,
            N=N_out,
            max_outer=max_outer,
            outer_tol=outer_tol,
            max_inner=max_inner,
            inner_tol=inner_tol,
            damping=damping,
            T=T
        )
        
        # ---------------------------------------------------------
        # 5. Store Results
        # ---------------------------------------------------------
        all_results[params['name']] = {
            'params': params,
            'baseline': {'k': best_k_base, 'i': best_i_base, 'EV': pp_EU_base},
            'temporal': {
                'taus': taus,
                'a_list': a_list,
                'b_list': b_list,
                'c': c,
                'ms': ms_out,
                'dx': dx_out,
                'N': N_out,
                'df_probs': df_probs,
                'L': L,
                'F': F,
                'EV': EV_dynamic,
                'i': best_i_dyn,
                'EU': best_EU,
                'i_search_results': i_search_results
            }
        }
        
        # ---------------------------------------------------------
        # Print Summary for this parameter set
        # ---------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {params['name']}")
        print(f"{'='*60}")
        print(f"\nParameters:")
        print(f"  beta = {beta}, sigma2 = {sigma2}, mu0 = {mu0}, tau0 = {tau0}, N = {N_total}")
        print(f"\nBaseline:")
        print(f"  Optimal (k, i) = ({best_k_base}, {best_i_base})")
        print(f"  Steady-state EV = {pp_EU_base:.6f}")
        print(f"\nTemporal-Aware:")
        print(f"  Expected lifetime L = {L:.4f}")
        print(f"  Steady-state EV = {EV_dynamic:.6f}")
        print(f"  Improvement factor = {EV_dynamic/pp_EU_base:.2f}x")
        print(f"{'='*60}")
    
    # ---------------------------------------------------------
    # 6. Create Comparative Plots
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("GENERATING COMPARATIVE PLOTS")
    print(f"{'='*60}\n")
        
    # Define colors for each parameter set
    colors = ['blue', 'red', 'green']
    
    # Plot 1: Decision Thresholds Comparison
    plt.figure(figsize=(10, 6))
    for idx, (name, results) in enumerate(all_results.items()):
        taus = results['temporal']['taus']
        a_list = results['temporal']['a_list']
        b_list = results['temporal']['b_list']
        t_vals = np.arange(1, len(taus) + 1)
        
        a_vals = np.array([np.nan if x is None else x for x in a_list])
        b_vals = np.array([np.nan if x is None else x for x in b_list])
        
        plt.plot(t_vals, a_vals, marker='o', markersize=3, label=f'{name} (shelve)', 
                linestyle='--', color=colors[idx], alpha=0.7)
        plt.plot(t_vals, b_vals, marker='s', markersize=3, label=f'{name} (ship)', 
                linestyle='-', color=colors[idx], alpha=0.7)
    
    plt.xlabel('Number of tests accumulated (per unit)', fontsize=11)
    plt.ylabel('Posterior mean threshold', fontsize=11)
    plt.title('Decision Thresholds: Comparison Across Parameter Sets', fontsize=12, fontweight='bold')
    plt.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.tight_layout()
    plt.savefig('Thresholds_comparison.png', dpi=150)
    print("Saved: Thresholds_comparison.png")
    plt.close()
    
    # Plot 2: Action Probabilities Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        df_probs = results['temporal']['df_probs']
        t_vals = df_probs["t"].values
        
        # Compute survival and cumulative probabilities
        S = np.ones_like(df_probs["Reject"])
        for i in range(1, len(S)):
            S[i] = S[i-1] * df_probs["Continue"].values[i-1]
        
        rej_occ = df_probs["Reject"].values * S
        buy_occ = df_probs["Buy"].values * S
        rej_cum = np.cumsum(rej_occ)
        buy_cum = np.cumsum(buy_occ)
        
        ax.plot(t_vals, df_probs["Reject"], label="Shelve", marker='o', markersize=3, color='red', alpha=0.7)
        ax.plot(t_vals, df_probs["Continue"], label="Continue", marker='s', markersize=3, color='blue', alpha=0.7)
        ax.plot(t_vals, df_probs["Buy"], label="Ship", marker='^', markersize=3, color='green', alpha=0.7)
        ax.plot(t_vals, rej_cum, label='Shelve (cumulative)', color='darkred', linestyle='--', alpha=0.6)
        ax.plot(t_vals, buy_cum, label='Ship (cumulative)', color='darkgreen', linestyle='--', alpha=0.6)
        
        ax.set_xlabel("Tests accumulated", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 2))
    
    fig.suptitle('Action Probabilities: Comparison Across Parameter Sets', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ActionP_comparison.png', dpi=150)
    print("Saved: ActionP_comparison.png")
    plt.close()
    
    # Plot 3: Performance Comparison Table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Parameter Set', 'Baseline EV', 'Sequential EV', 'Improvement', 'Lifetime L', 'Throughput F']]
    for name, results in all_results.items():
        baseline_ev = results['baseline']['EV']
        sequential_ev = results['temporal']['EV']
        improvement = sequential_ev / baseline_ev
        lifetime = results['temporal']['L']
        throughput = results['temporal']['F']
        table_data.append([
            name, 
            f"{baseline_ev:.6f}", 
            f"{sequential_ev:.6f}", 
            f"{improvement:.2f}x",
            f"{lifetime:.2f}",
            f"{throughput:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Performance Comparison: Baseline vs Sequential Bayesian Testing', 
             fontsize=12, fontweight='bold', pad=20)
    plt.savefig('Performance_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: Performance_comparison.png")
    plt.close()
    
    # Plot 4: Optimal j (i) Search Comparison
    plt.figure(figsize=(10, 6))
    for idx, (name, results) in enumerate(all_results.items()):
        i_search_results = results['temporal']['i_search_results']
        i_vals = [r[0] for r in i_search_results]
        EV_vals = [r[4] for r in i_search_results]
        best_i = results['temporal']['i']
        
        plt.plot(i_vals, EV_vals, label=name, color=colors[idx], alpha=0.7)
        plt.axvline(best_i, color=colors[idx], linestyle='--', alpha=0.5, 
                   label=f'{name} optimal j={best_i}')
    
    plt.xlabel('Number of ideas tested per period', fontsize=11)
    plt.ylabel('Steady-state expected value', fontsize=11)
    plt.title('Optimal Testing Capacity Allocation: Comparison Across Parameter Sets', 
             fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.tight_layout()
    plt.savefig('Optimal_j_comparison.png', dpi=150)
    print("Saved: Optimal_j_comparison.png")
    plt.close()
    
    # Plot 5: Value Gains from Sequential Testing 
    plt.figure(figsize=(10, 6))
    
    # Plot the naive baseline
    ms_ref = list(all_results.values())[0]['temporal']['ms']
    V_naive = np.maximum(ms_ref, 0.0)
    plt.plot(ms_ref, V_naive, label='Naive baseline', color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    for idx, (name, results) in enumerate(all_results.items()):
        # Get data for this parameter set
        ms = results['temporal']['ms']
        taus = results['temporal']['taus']
        a_list = results['temporal']['a_list']
        b_list = results['temporal']['b_list']
        c = results['temporal']['c']
        beta_param = results['params']['beta']
        sigma2 = results['params']['sigma2']
        
        # Get tau1 and recompute V for plotting
        tau1 = taus[0]
        from Temporal_aware import solve_value_iteration_for_tau, build_transition_kernel, update_tau
        tau2 = update_tau(tau1, sigma2)
        var_mprime1 = max(tau1 - tau2, 0.0)
        N_grid = len(ms)
        dx_grid = (ms[-1] - ms[0]) / (N_grid - 1)
        P1 = build_transition_kernel(ms, var_mprime1, N_grid, dx_grid)
        
        V_temporal, _, _ = solve_value_iteration_for_tau(
            c, tau1, ms, sigma2, beta_param, 
            max_inner=5000, inner_tol=1e-9, N=N_grid, dx=dx_grid, P=P1, verbose=False
        )
        
        # Plot temporal-aware with matching colors from thresholds_comparison
        plt.plot(ms, V_temporal, label=f'{name} (temporal-aware)', color=colors[idx], linewidth=2, alpha=0.7)
    
    plt.xlabel('Posterior mean', fontsize=11)
    plt.ylabel('Expected return', fontsize=11)
    plt.title('Value Gains from Sequential Testing vs Naive Baseline', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('Value_gains_comparison.png', dpi=150)
    print("Saved: Value_gains_comparison.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
