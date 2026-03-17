import os
import pandas as pd
import numpy as np
import glob
import json
from evaluation.plotting import COLORS, plt, plot_bar_comparison, plot_perclass_bar

def plot_convergence_from_history(precision):
    """Filters history files from a single directory based on filename type."""
    history_files = glob.glob("outputs/history/history_*.json")

    target_type = f"type{precision.lower()}"
    filtered_files = [f for f in history_files if target_type in f]
    
    if not filtered_files:
        print(f"[SKIP] No history files found for {precision} (Looking for '{target_type}')")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    STYLE_MAP = {
        ("VAE", "0.5"):  {"color": COLORS[0], "ls": "-",  "label": "VAE (α=0.5)"},
        ("VAE", "1.0"):  {"color": COLORS[0], "ls": "--", "label": "VAE (α=1.0)"},
        ("CONV", "0.5"): {"color": COLORS[1], "ls": "-",  "label": "CONV (α=0.5)"},
        ("CONV", "1.0"): {"color": COLORS[1], "ls": "--", "label": "CONV (α=1.0)"},
    }
    
    grouped = {}
    for h_file in filtered_files:
        with open(h_file, 'r') as f:
            data = json.load(f)
            fname = os.path.basename(h_file)
            try:
                model = fname.split('_')[1].upper()
                alpha = fname.split('alpha')[1].split('_')[0]
                key = (model, alpha)
                if key not in grouped: grouped[key] = []
                grouped[key].append(data['loss'])
            except IndexError:
                continue

    for (model, alpha), loss_lists in grouped.items():
        min_len = min(len(l) for l in loss_lists)
        avg_loss = np.mean([l[:min_len] for l in loss_lists], axis=0)
        style = STYLE_MAP.get((model, alpha), {"color": "k", "ls": "-", "label": f"{model} {alpha}"})
        
        ax.plot(range(1, min_len + 1), avg_loss, 
                label=style["label"], color=style["color"], 
                linestyle=style["ls"], lw=2.0, alpha=0.8)

    ax.set_xlabel("Round")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Training Convergence Curve ({precision})")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle=':', alpha=0.6)
    
    save_path = f"outputs/figures/convergence_{precision.lower()}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[DONE] {save_path} saved.")

def plot_robustness_with_shared_method(precision):
    """Filters a single CSV by the 'precision_type' column."""
    csv_path = "outputs/final_results.csv"
    if not os.path.exists(csv_path):
        return

    full_df = pd.read_csv(csv_path)
    df = full_df[full_df['precision_type'].str.lower() == precision.lower()].copy()
    
    if df.empty:
        print(f"[SKIP] No data in CSV for precision: {precision}")
        return

    df['auroc'] = pd.to_numeric(df['auroc'], errors='coerce')
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')

    target_models = ['vae', 'conv']
    target_alphas = [0.5, 1.0]
    metrics = ["ALPHA 0.5\n(CLEAN)", "ALPHA 0.5\n(DP-SGD)", 
               "ALPHA 1.0\n(CLEAN)", "ALPHA 1.0\n(DP-SGD)"]
    
    formatted_data = {}
    for model in target_models:
        m_name = model.upper()
        formatted_data[m_name] = {}
        for alpha in target_alphas:
            clean_val = df[(df['model'] == model) & (df['alpha'] == alpha) & (df['epsilon'] == float('inf'))]['auroc'].mean()
            dp_vals = df[(df['model'] == model) & (df['alpha'] == alpha) & (df['epsilon'] < float('inf'))]['auroc'].dropna()
            
            formatted_data[m_name][f"ALPHA {alpha}\n(CLEAN)"] = {"mean": clean_val if not np.isnan(clean_val) else 0.0, "std": 0}
            formatted_data[m_name][f"ALPHA {alpha}\n(DP-SGD)"] = {"mean": dp_vals.mean() if not dp_vals.empty else 0.0, "std": 0}

    save_path = f"outputs/figures/non_iid_vs_auroc_{precision.lower()}.png"
    plot_bar_comparison(data=formatted_data, metrics=metrics, title=f"Non-IID (Dirichlet α) vs AUROC ({precision})", save_path=save_path)

    ax = plt.gca()
    plt.xticks(rotation=0, fontsize=8)
    ax.legend(title="Settings", loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.2)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.3f}', (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 5), textcoords="offset points")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[DONE] {save_path} saved.")

def plot_perclass_with_shared_method(precision):
    """Filters a single CSV by 'precision_type' for per-class AUROC."""
    csv_path = "outputs/final_results.csv"
    if not os.path.exists(csv_path): return

    full_df = pd.read_csv(csv_path)
    df = full_df[full_df['precision_type'].str.lower() == precision.lower()].copy()
    if df.empty: return

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
                perclass_data[m_name][cond] = {"auroc": {"mean": vals.mean() if not vals.empty else 0.0, "std": 0}}

    save_path = f"outputs/figures/per_class_{precision.lower()}.png"
    plot_perclass_bar(perclass_data=perclass_data, metric="auroc", conditions=conditions, save_path=save_path)

    ax = plt.gca()
    ax.legend(title="Models", loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.2)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.3f}', (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords="offset points")

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[DONE] {save_path} saved.")

if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    for precision in ["FP32", "INT8"]:
        print(f"\n--- Processing: {precision} ---")
        plot_convergence_from_history(precision)
        plot_robustness_with_shared_method(precision)
        plot_perclass_with_shared_method(precision)