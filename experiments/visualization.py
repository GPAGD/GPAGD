import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_convergence(loss_curves, bench_name, save_path=None):
    """Plot loss curves for a given benchmark."""
    plt.figure()
    for label, loss_hist in loss_curves.items():
        plt.plot(loss_hist, label=label, linewidth=1.5)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(bench_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_bar_chart(results, benchmarks, variants, save_path=None):
    """Create ablation bar chart."""
    x = np.arange(len(benchmarks))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, var in enumerate(variants):
        means = [np.mean(results[bench][var]) for bench in benchmarks]
        stds = [np.std(results[bench][var]) for bench in benchmarks]
        ax.bar(x + i*width, means, width, yerr=stds, label=var, capsize=3)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel('Relative L2 Error')
    ax.set_yscale('log')
    ax.set_title('Ablation Study: GPAGD vs Adam')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def save_results_to_csv(results, benchmarks, variants, save_path):
    """Save mean and std of relative L2 errors to CSV."""
    rows = []
    for bench in benchmarks:
        for var in variants:
            rel_errors = results[bench][var]
            rows.append({
                'Benchmark': bench,
                'Variant': var,
                'Mean_Rel_L2': np.mean(rel_errors),
                'Std_Rel_L2': np.std(rel_errors)
            })
    pd.DataFrame(rows).to_csv(save_path, index=False)
