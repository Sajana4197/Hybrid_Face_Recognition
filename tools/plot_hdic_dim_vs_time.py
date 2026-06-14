"""
Figure 2: Effect of hypervector dimensionality on average HDIC hash generation time.
Runs the full benchmark 5 times, averages the results, then plots.

Run from repo root:
    python tools/plot_hdic_dim_vs_time.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt

import hdic.encode_hv as encode_hv_module
from hdic.encode_hv import encode_embedding_to_hv

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_TRIALS     = 500   # trials per dimension per run
N_RUNS       = 5     # number of full benchmark runs to average over

HV_DIMS = [
    5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,
    10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 
    15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000,
    20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000,
    25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000
]

DIM_ORIG = encode_hv_module.DIM_ORIG  # 512

# ── Benchmark: N_RUNS full sweeps ───────────────���─────────────────────────────
# all_run_avgs shape: (N_RUNS, len(HV_DIMS))
all_run_avgs = np.zeros((N_RUNS, len(HV_DIMS)))

for run in range(N_RUNS):
    print(f"\n{'='*50}")
    print(f"  Run {run + 1} / {N_RUNS}")
    print(f"{'='*50}")
    print(f"{'DIM_HV':>8}  {'Avg (ms)':>12}  {'Std (ms)':>12}")
    print("-" * 38)

    rng = np.random.default_rng(RANDOM_SEED + run)  # different seed each run

    for i, dim_hv in enumerate(HV_DIMS):
        np.random.seed(RANDOM_SEED)
        encode_hv_module.projection_matrix = np.random.randn(DIM_ORIG, dim_hv)

        times = []
        for _ in range(N_TRIALS):
            embedding = rng.standard_normal(DIM_ORIG).astype(np.float32)
            t_start = time.perf_counter()
            _ = encode_embedding_to_hv(embedding)
            t_end   = time.perf_counter()
            times.append(t_end - t_start)

        avg_ms = np.mean(times) * 1_000
        std_ms = np.std(times)  * 1_000
        all_run_avgs[run, i] = avg_ms

        print(f"{dim_hv:>8}  {avg_ms:>12.4f}  {std_ms:>12.4f}")

# Restore original projection matrix
np.random.seed(RANDOM_SEED)
encode_hv_module.projection_matrix = np.random.randn(DIM_ORIG, encode_hv_module.DIM_HV)

# ── Compute final average and std across runs ─────────────────────────────────
final_avg = np.mean(all_run_avgs, axis=0)   # mean across N_RUNS
final_std = np.std(all_run_avgs, axis=0)    # std across N_RUNS

print(f"\n{'='*50}")
print(f"  Final Average across {N_RUNS} runs")
print(f"{'='*50}")
print(f"{'DIM_HV':>8}  {'Mean (ms)':>12}  {'Std (ms)':>12}")
print("-" * 38)
for dim_hv, avg, std in zip(HV_DIMS, final_avg, final_std):
    print(f"{dim_hv:>8}  {avg:>12.4f}  {std:>12.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# Shaded band = std across the 5 runs
ax.fill_between(HV_DIMS,
                final_avg - final_std,
                final_avg + final_std,
                alpha=0.15, color='steelblue', label='±1 std dev (across runs)')

# One line per run (light, for transparency)
for run in range(N_RUNS):
    ax.plot(HV_DIMS, all_run_avgs[run],
            color='steelblue', linewidth=0.7, alpha=0.3,
            linestyle='--')

# Final averaged line on top
ax.plot(HV_DIMS, final_avg,
        color='steelblue', linewidth=2.2,
        marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.8,
        label=f'Mean avg time ({N_RUNS} runs)')

ax.set_xlabel('Number of Dimensions (D)', fontsize=12)
ax.set_ylabel('Average Hash Generation Time (ms)', fontsize=12)
ax.set_title('Effect of Hypervector Dimensionality on\nAverage HDIC Hash Generation Time',
             fontsize=13, fontweight='bold')
ax.set_xlim(HV_DIMS[0] - 200, HV_DIMS[-1] + 500)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.tick_params(axis='both', labelsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure2_hdic_dim_vs_time.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nFigure 2 saved → {out_path}")