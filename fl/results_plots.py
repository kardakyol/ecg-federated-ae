import os
import pandas as pd
import numpy as np
import glob
import json
from evaluation.plotting import COLORS, plt, plot_bar_comparison, plot_perclass_bar

def plot_convergence_from_history():
    """Custom logic for convergence since plotting.py doesn't have a specific history method."""
    history_files = glob.glob("outputs/history/history_*.json")
    if not history_files:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    MODEL_COLORS = {"VAE": COLORS[0], "CONV": COLORS[1]}
    
    grouped = {}
    for h_file in history_files:
        with open(h_file, 'r') as f:
            data = json.load(f)
            fname = os.path.basename(h_file)
            model = fname.split('_')[1].upper()
            alpha = fname.split('alpha')[1].split('_')[0]
            key = (model, alpha)
            if key not in grouped: grouped[key] = []
            grouped[key].append(data['loss'])

    for (model, alpha), loss_lists in grouped.items():
        min_len = min(len(l) for l in loss_lists)
        avg_loss = np.mean([l[:min_len] for l in loss_lists], axis=0)
        ax.plot(range(1, min_len + 1), avg_loss, 
                label=f"{model} ($\\alpha$={alpha})", 
                color=MODEL_COLORS.get(model, "#000000"), lw=1.5)

    ax.set(xlabel="Round", ylabel="MSE", title="Global Convergence")
    ax.legend(loc="upper right")
    fig.savefig("outputs/figures/convergence_curves.png")
    print("[DONE] convergence_curves.png saved.")

def plot_robustness_with_shared_method():
    """FIX: Used consistent UPPERCASE keys to prevent KeyError and LaTeX errors."""
    csv_path = "outputs/dp_fl_results.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['auroc'] = pd.to_numeric(df['auroc'], errors='coerce')
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')

    target_models = ['vae', 'conv']
    target_alphas = [0.5, 1.0]
    
    # --- FIX: Match keys exactly and avoid LaTeX symbols to prevent ParseErrors ---
    metrics = [
        "ALPHA 0.5\n(CLEAN)", "ALPHA 0.5\n(DP-SGD)",
        "ALPHA 1.0\n(CLEAN)", "ALPHA 1.0\n(DP-SGD)"
    ]
    
    formatted_data = {}
    for model in target_models:
        m_name = model.upper()
        formatted_data[m_name] = {}
        for alpha in target_alphas:
            # Filter Clean (inf)
            clean_val = df[(df['model'] == model) & (df['alpha'] == alpha) & (df['epsilon'] == float('inf'))]['auroc'].mean()
            # Filter Private (DP)
            dp_vals = df[(df['model'] == model) & (df['alpha'] == alpha) & (df['epsilon'] < float('inf'))]['auroc'].dropna()
            
            # --- FIX: Dictionary keys MUST match the 'metrics' list exactly ---
            formatted_data[m_name][f"ALPHA {alpha}\n(CLEAN)"] = {
                "mean": clean_val if not np.isnan(clean_val) else 0.0,
                "std": 0  # FIX: Set to 0 to remove the lines at the top
            }
            formatted_data[m_name][f"ALPHA {alpha}\n(DP-SGD)"] = {
                "mean": dp_vals.mean() if not dp_vals.empty else 0.0,
                "std": 0
            }

    if formatted_data:
        save_path = "outputs/figures/non_iid_vs_auroc.png"
        plot_bar_comparison(
            data=formatted_data,
            metrics=metrics, 
            title="Privacy-Utility Tradeoff by Data Heterogeneity",
            save_path=save_path
        )

        # --- FIX: Annotate and Legend (Post-processing) ---
        ax = plt.gca()
        plt.xticks(rotation=0, fontsize=8)
        ax.legend(title="Settings", loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.15)
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f'{val:.3f}', (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            xytext=(0, 5), textcoords="offset points")
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[DONE] {save_path} saved.")

def plot_perclass_with_shared_method():
    """FIX: Added value printing and moved legend outside."""
    df = pd.read_csv("outputs/dp_fl_results.csv")
    conditions = ["MI", "STTC", "HYP", "CD"]
    
    perclass_data = {}
    for model in df['model'].unique():
        m_name = model.upper()
        perclass_data[m_name] = {}
        m_df = df[df['model'] == model]
        for cond in conditions:
            col = f"{cond}_auroc"
            if col in df.columns:
                vals = pd.to_numeric(m_df[col], errors='coerce').dropna()
                perclass_data[m_name][cond] = {
                    "auroc": {"mean": vals.mean() if not vals.empty else 0.0, "std": 0}
                }

    if perclass_data:
        save_path = "outputs/figures/per_class_comparison.png"
        plot_perclass_bar(
            perclass_data=perclass_data,
            metric="auroc",
            conditions=conditions,
            save_path=save_path
        )

        # --- FIX: Annotate and Legend ---
        ax = plt.gca()
        plt.xticks(fontsize=9)
        ax.legend(title="Models", loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.15)
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f'{val:.3f}', (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            xytext=(0, 3), textcoords="offset points")

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[DONE] {save_path} saved.")

if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    plot_convergence_from_history()
    plot_robustness_with_shared_method()
    plot_perclass_with_shared_method()