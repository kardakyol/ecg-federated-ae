"""
Fig 3 - ROC Curves Overlay (3 AEs x FP32 + INT8)
Fig 4 - Pareto Front (AUROC vs Model Size & FLOPs) - FIXED
IEEE-ready, 300 DPI PDF output
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

roc_df   = pd.read_csv('outputs/roc_scores.csv')
auroc_df = pd.read_csv('outputs/auroc_degradation.csv')

auroc = {
    row['model']: {'fp32': row['fp32_auroc'], 'int8': row['int8_auroc']}
    for _, row in auroc_df.iterrows()
}

MODEL_NAMES = {
    'vanilla_ae': 'VanillaAE',
    'conv_ae':    'ConvAE',
    'vae':        'VAE',
}

COLORS = {
    'vanilla_ae': '#378ADD',
    'conv_ae':    '#1D9E75',
    'vae':        '#D85A30',
}

MODELS = ['vanilla_ae', 'conv_ae', 'vae']

os.makedirs('outputs/figures', exist_ok=True)

# ── Fig 3: ROC Curves ─────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(5.5, 4.5))

for model in MODELS:
    color = COLORS[model]
    name  = MODEL_NAMES[model]
    sub_fp32 = roc_df[(roc_df['model']==model) & (roc_df['precision_type']=='fp32')]
    ax.plot(sub_fp32['fpr'], sub_fp32['tpr'],
            color=color, linewidth=1.8, linestyle='-',
            label=f'{name} FP32 (AUC={auroc[model]["fp32"]:.4f})')
    sub_int8 = roc_df[(roc_df['model']==model) & (roc_df['precision_type']=='int8')]
    ax.plot(sub_int8['fpr'], sub_int8['tpr'],
            color=color, linewidth=1.2, linestyle='--',
            label=f'{name} INT8 (AUC={auroc[model]["int8"]:.4f})')

ax.plot([0,1],[0,1], color='#888780', linewidth=0.8,
        linestyle=':', label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curves - FP32 vs INT8 Quantisation', fontsize=10, fontweight='normal')
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1])
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=9)
ax.text(0.38, 0.08,
        'Max AUROC degradation < 0.12%\nacross all models',
        fontsize=7.5, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1EFE8',
                  edgecolor='#B4B2A9', alpha=0.9))

fig3.tight_layout()
fig3.savefig('outputs/figures/fig3_roc_curves.pdf', dpi=300, bbox_inches='tight')
fig3.savefig('outputs/figures/fig3_roc_curves.png', dpi=300, bbox_inches='tight')
print("Fig 3 saved.")

# ── Fig 4: Pareto Front (FIXED) ───────────────────────────────────────────────
pareto_data = {
    'vanilla_ae': {'size_fp32':48.0741,'size_int8':12.0679,'flops':25.21,
                   'auroc_fp32':0.6369,'auroc_int8':0.6368},
    'conv_ae':    {'size_fp32':5.7112, 'size_int8':2.7597, 'flops':84.98,
                   'auroc_fp32':0.7876,'auroc_int8':0.7869},
    'vae':        {'size_fp32':8.3012, 'size_int8':3.8738, 'flops':110.21,
                   'auroc_fp32':0.7945,'auroc_int8':0.7936},
}

fig4, axes = plt.subplots(1, 2, figsize=(9, 4))

# ── (a) AUROC vs Model Size ────────────────────────────────────────────────────
ax1 = axes[0]
for model in MODELS:
    d     = pareto_data[model]
    color = COLORS[model]
    name  = MODEL_NAMES[model]

    ax1.scatter(d['size_fp32'], d['auroc_fp32'],
                color=color, marker='o', s=90, zorder=5)
    ax1.scatter(d['size_int8'], d['auroc_int8'],
                color=color, marker='s', s=90, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax1.annotate('', xy=(d['size_int8'], d['auroc_int8']),
                 xytext=(d['size_fp32'], d['auroc_fp32']),
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Labels next to FP32 point
    offset_x = {'vanilla_ae': 1.0, 'conv_ae': 0.5, 'vae': 0.5}
    offset_y = {'vanilla_ae': 0.008, 'conv_ae': 0.007, 'vae': -0.014}
    ax1.text(d['size_fp32'] + offset_x[model],
             d['auroc_fp32'] + offset_y[model],
             name, fontsize=8.5, color=color)

ax1.annotate('Pareto-optimal\n(ConvAE INT8)',
             xy=(2.7597, 0.7869), xytext=(12, 0.73), fontsize=7.5,
             arrowprops=dict(arrowstyle='->', color='#444441', lw=0.9),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1F5EE',
                       edgecolor='#1D9E75', alpha=0.85))

ax1.set_xlabel('Model Size (MB)', fontsize=10)
ax1.set_ylabel('AUROC', fontsize=10)
ax1.set_title('(a) AUROC vs Model Size', fontsize=10, fontweight='normal')
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.tick_params(labelsize=9)
ax1.set_xlim([-2, 54])      # FIXED: start from 0 so VanillaAE arrow is visible
ax1.set_ylim([0.60, 0.83])

# ── (b) AUROC vs FLOPs ────────────────────────────────────────────────────────
ax2 = axes[1]
JITTER = 1.8   # horizontal offset to separate FP32 and INT8 dots

for model in MODELS:
    d     = pareto_data[model]
    color = COLORS[model]
    name  = MODEL_NAMES[model]

    # FP32 slightly left, INT8 slightly right
    ax2.scatter(d['flops'] - JITTER, d['auroc_fp32'],
                color=color, marker='o', s=90, zorder=5)
    ax2.scatter(d['flops'] + JITTER, d['auroc_int8'],
                color=color, marker='s', s=90, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax2.annotate('', xy=(d['flops'] + JITTER, d['auroc_int8']),
                 xytext=(d['flops'] - JITTER, d['auroc_fp32']),
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    offset_x2 = {'vanilla_ae': -14, 'conv_ae': 2.5, 'vae': 2.5}
    offset_y2 = {'vanilla_ae': 0.007, 'conv_ae': 0.007, 'vae': -0.014}
    ax2.text(d['flops'] + offset_x2[model],
             d['auroc_fp32'] + offset_y2[model],
             name, fontsize=8.5, color=color)

ax2.set_xlabel('FLOPs (M)', fontsize=10)
ax2.set_ylabel('AUROC', fontsize=10)
ax2.set_title('(b) AUROC vs FLOPs', fontsize=10, fontweight='normal')
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.tick_params(labelsize=9)
ax2.set_xlim([15, 120])
ax2.set_ylim([0.60, 0.83])

# Shared legend
legend_elements = [
    Line2D([0],[0], marker='o', color='#888780', markersize=7,
           label='FP32', linewidth=0),
    Line2D([0],[0], marker='s', color='#888780', markersize=7,
           markeredgecolor='white', label='INT8', linewidth=0),
]
for model in MODELS:
    legend_elements.append(
        Line2D([0],[0], color=COLORS[model], linewidth=2,
               label=MODEL_NAMES[model])
    )

fig4.legend(handles=legend_elements, loc='lower center',
            ncol=5, fontsize=8, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.08))
fig4.suptitle('Pareto Front - AUROC vs Computational Cost',
              fontsize=11, fontweight='normal', y=1.01)
fig4.tight_layout()
fig4.savefig('outputs/figures/fig4_pareto_front.pdf', dpi=300, bbox_inches='tight')
fig4.savefig('outputs/figures/fig4_pareto_front.png', dpi=300, bbox_inches='tight')
print("Fig 4 saved.")
print("Done.")
