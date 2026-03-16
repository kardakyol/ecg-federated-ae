import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os

def plot_results():
    os.makedirs("outputs/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")

    history_files = glob.glob("outputs/history/history_*.json")
    if history_files:
        plt.figure(figsize=(10, 6))
        for h_file in history_files:
            with open(h_file, 'r') as f:
                data = json.load(f)
                fname = os.path.basename(h_file)
                model_name = fname.split('_')[1].upper()
                alpha_val = fname.split('alpha')[1].split('_')[0]
                plt.plot(data['rounds'], data['loss'], label=f"{model_name} (α={alpha_val})", linewidth=2)
        
        plt.title("Training convergence curves (round vs loss per AE)", fontsize=14)
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel("MSE (Log Scale)", fontsize=12)	
        plt.yscale('log')
        plt.legend()
        plt.savefig("outputs/figures/convergence_curves.png", dpi=300)
        print("[DONE] convergence_curves.png saved.")

    if os.path.exists("outputs/dp_fl_results.csv"):
        df = pd.read_csv("outputs/dp_fl_results.csv")
        plot_df = df[(df['model'] == 'vae') & (df['epsilon'] == 10.0)].sort_values('alpha')
        
        if not plot_df.empty:
            plt.figure(figsize=(8, 5))
            ax = sns.barplot(data=plot_df, x='alpha', y='auroc', palette="magma")
            plt.title("Model Robustness to Data Heterogeneity (Dirichlet α vs. AUROC)", fontsize=14)
            plt.xlabel("Heterogeneity Level (Dirichlet α)", fontsize=12)
            plt.ylabel("AUROC", fontsize=12)
            plt.ylim(0.5, 0.8)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold')    
            plt.savefig("outputs/figures/non_iid_robustness.png", dpi=300)
            print("[DONE] non_iid_robustness.png saved.")

if __name__ == "__main__":
    plot_results()
	