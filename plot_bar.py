"""
Bar Chart - Ghadah Sprint 4
Run: python plot_bar.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

os.makedirs('outputs/figures', exist_ok=True)

mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

pc  = pd.read_csv('outputs/quantisation_results.csv')
pi4 = pd.read_csv('outputs/pi4_results.csv').drop_duplicates(
        subset=['model','precision_type','seed'])
auroc = pd.read_csv('outputs/auroc_degradation.csv')

models = ['vanilla_ae', 'conv_ae', 'vae']
labels = ['VanillaAE', 'ConvAE', 'VAE']
x = np.arange(len(models))
w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor='white')
fig.subplots_adjust(wspace=0.35)

# Panel 1: Model Size
ax = axes[0]
fp32_vals = [pc[(pc.model==m)&(pc.precision_type=='fp32')]['model_size_mb'].mean() for m in models]
int8_vals = [pc[(pc.model==m)&(pc.precision_type=='int8')]['model_size_mb'].mean() for m in models]
ax.bar(x-w/2, fp32_vals, w, label='FP32', color='#1f77b4', edgecolor='white', linewidth=0.5)
ax.bar(x+w/2, int8_vals, w, label='INT8', color='#2ca02c', edgecolor='white', linewidth=0.5)
reductions = [74.9, 51.7, 53.3]
for i, (fp_h, int_h, r) in enumerate(zip(fp32_vals, int8_vals, reductions)):
    y = fp_h + 2.0
    ax.text(x[i]-w/2, y, f'-{r}%',
            ha='center', fontsize=10, color='#d62728', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Model Size (MB)')
ax.set_title('(a) Model Size Reduction', fontweight='bold', pad=10)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 62)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# Panel 2: Pi4 Latency
ax = axes[1]
fp32_lat = [pi4[(pi4.model==m)&(pi4.precision_type=='fp32')]['inference_latency_ms'].mean() for m in models]
int8_lat = [pi4[(pi4.model==m)&(pi4.precision_type=='int8')]['inference_latency_ms'].mean() for m in models]
fp32_std = [pi4[(pi4.model==m)&(pi4.precision_type=='fp32')]['inference_latency_ms'].std() for m in models]
int8_std = [pi4[(pi4.model==m)&(pi4.precision_type=='int8')]['inference_latency_ms'].std() for m in models]
ax.bar(x-w/2, fp32_lat, w, label='FP32', color='#1f77b4',
       yerr=fp32_std, capsize=4, error_kw={'ecolor':'#555','linewidth':1.2},
       edgecolor='white', linewidth=0.5)
ax.bar(x+w/2, int8_lat, w, label='INT8', color='#2ca02c',
       yerr=int8_std, capsize=4, error_kw={'ecolor':'#555','linewidth':1.2},
       edgecolor='white', linewidth=0.5)
speedups = [1.80, 1.20, 1.26]
for xi, sp, f, t in zip(x, speedups, fp32_lat, int8_lat):
    ax.text(xi, max(f,t)+2.5, f'x{sp}',
            ha='center', fontsize=10, color='#d62728', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Inference Latency (ms)')
ax.set_title('(b) Pi4 Inference Latency', fontweight='bold', pad=10)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 58)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# Panel 3: AUROC
ax = axes[2]
fp32_auroc = [auroc[auroc.model==m]['fp32_auroc'].values[0] for m in models]
int8_auroc = [auroc[auroc.model==m]['int8_auroc'].values[0] for m in models]
ax.bar(x-w/2, fp32_auroc, w, label='FP32', color='#1f77b4', edgecolor='white', linewidth=0.5)
ax.bar(x+w/2, int8_auroc, w, label='INT8', color='#2ca02c', edgecolor='white', linewidth=0.5)
for i, (f, t) in enumerate(zip(fp32_auroc, int8_auroc)):
    ax.text(x[i]-w/2 - 0.1, f+0.004, f'{f:.4f}',
            ha='center', fontsize=8.5, color='#1f77b4', fontweight='bold')
    ax.text(x[i]+w/2 + 0.1, t+0.004, f'{t:.4f}',
            ha='center', fontsize=8.5, color='#2ca02c', fontweight='bold')
ax.set_ylim(0.55, 0.92)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('AUROC')
ax.set_title('(c) AUROC Degradation', fontweight='bold', pad=10)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.text(0.5, 0.06, 'DELTA AUROC < 0.12% across all models',
        transform=ax.transAxes, ha='center', fontsize=10,
        color='#d62728', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3f3', alpha=0.8))
for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.suptitle('INT8 Post-Training Quantisation - ECG Federated Autoencoder',
             fontsize=14, fontweight='bold', y=1.03)
plt.savefig('outputs/figures/bar_chart.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('outputs/figures/bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved!")
