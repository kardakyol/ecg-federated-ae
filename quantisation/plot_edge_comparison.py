"""
Fig 8 - Edge Device Comparison: PC vs Raspberry Pi 4
FP32 and INT8 latency comparison across all models
IEEE-ready, 300 DPI PDF
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs('outputs/figures', exist_ok=True)

# ── Real data (mean over seeds 42, 123, 456) ──────────────────────────────────
data = {
    'vanilla_ae': {
        'pc_fp32':  2.188, 'pc_fp32_std':  0.339,
        'pc_int8':  1.502, 'pc_int8_std':  0.108,
        'pi4_fp32': 45.152,'pi4_fp32_std': 0.173,
        'pi4_int8': 25.145,'pi4_int8_std': 0.107,
    },
    'conv_ae': {
        'pc_fp32':  2.657, 'pc_fp32_std':  0.658,
        'pc_int8':  2.905, 'pc_int8_std':  0.457,
        'pi4_fp32': 16.908,'pi4_fp32_std': 0.132,
        'pi4_int8': 14.097,'pi4_int8_std': 0.076,
    },
    'vae': {
        'pc_fp32':  3.162, 'pc_fp32_std':  0.946,
        'pc_int8':  4.412, 'pc_int8_std':  1.577,
        'pi4_fp32': 21.005,'pi4_fp32_std': 0.201,
        'pi4_int8': 16.639,'pi4_int8_std': 0.010,
    },
}

MODELS      = ['vanilla_ae', 'conv_ae', 'vae']
MODEL_NAMES = {'vanilla_ae': 'VanillaAE', 'conv_ae': 'ConvAE', 'vae': 'VAE'}

# Slowdown factors (Pi4/PC)
slowdown = {
    'vanilla_ae': {'fp32': 45.152/2.188, 'int8': 25.145/1.502},
    'conv_ae':    {'fp32': 16.908/2.657, 'int8': 14.097/2.905},
    'vae':        {'fp32': 21.005/3.162, 'int8': 16.639/4.412},
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ── (a) FP32 comparison ───────────────────────────────────────────────────────
ax1 = axes[0]
x     = np.arange(len(MODELS))
width = 0.35

bars_pc  = ax1.bar(x - width/2,
                   [data[m]['pc_fp32']  for m in MODELS],
                   width, yerr=[data[m]['pc_fp32_std'] for m in MODELS],
                   label='PC (x86-64)', color='#378ADD',
                   capsize=4, error_kw={'linewidth':1.0})
bars_pi4 = ax1.bar(x + width/2,
                   [data[m]['pi4_fp32'] for m in MODELS],
                   width, yerr=[data[m]['pi4_fp32_std'] for m in MODELS],
                   label='Raspberry Pi 4', color='#D85A30',
                   capsize=4, error_kw={'linewidth':1.0})

# Value labels
for i, model in enumerate(MODELS):
    ax1.text(i - width/2, data[model]['pc_fp32'] + data[model]['pc_fp32_std'] + 0.4,
             f"{data[model]['pc_fp32']:.1f}", ha='center', fontsize=7.5, color='#378ADD')
    ax1.text(i + width/2, data[model]['pi4_fp32'] + data[model]['pi4_fp32_std'] + 0.4,
             f"{data[model]['pi4_fp32']:.1f}", ha='center', fontsize=7.5, color='#D85A30')
    # Slowdown annotation
    ax1.text(i, data[model]['pi4_fp32']/2,
             f"×{slowdown[model]['fp32']:.1f}", ha='center', fontsize=8,
             color='white', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels([MODEL_NAMES[m] for m in MODELS], fontsize=10)
ax1.set_ylabel('Inference Latency (ms)', fontsize=10)
ax1.set_title('(a) FP32 Latency: PC vs Pi4', fontsize=10, fontweight='normal')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.25, linewidth=0.5, axis='y')
ax1.tick_params(labelsize=9)
ax1.set_ylim([0, 52])

# ── (b) INT8 comparison ───────────────────────────────────────────────────────
ax2 = axes[1]

bars_pc2  = ax2.bar(x - width/2,
                    [data[m]['pc_int8']  for m in MODELS],
                    width, yerr=[data[m]['pc_int8_std'] for m in MODELS],
                    label='PC (x86-64)', color='#378ADD',
                    capsize=4, error_kw={'linewidth':1.0},
                    hatch='//')
bars_pi42 = ax2.bar(x + width/2,
                    [data[m]['pi4_int8'] for m in MODELS],
                    width, yerr=[data[m]['pi4_int8_std'] for m in MODELS],
                    label='Raspberry Pi 4', color='#D85A30',
                    capsize=4, error_kw={'linewidth':1.0},
                    hatch='//')

# Value labels
for i, model in enumerate(MODELS):
    ax2.text(i - width/2, data[model]['pc_int8'] + data[model]['pc_int8_std'] + 0.4,
             f"{data[model]['pc_int8']:.1f}", ha='center', fontsize=7.5, color='#378ADD')
    ax2.text(i + width/2, data[model]['pi4_int8'] + data[model]['pi4_int8_std'] + 0.4,
             f"{data[model]['pi4_int8']:.1f}", ha='center', fontsize=7.5, color='#D85A30')
    # Slowdown annotation
    ax2.text(i, data[model]['pi4_int8']/2,
             f"×{slowdown[model]['int8']:.1f}", ha='center', fontsize=8,
             color='white', fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels([MODEL_NAMES[m] for m in MODELS], fontsize=10)
ax2.set_ylabel('Inference Latency (ms)', fontsize=10)
ax2.set_title('(b) INT8 Latency: PC vs Pi4', fontsize=10, fontweight='normal')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.25, linewidth=0.5, axis='y')
ax2.tick_params(labelsize=9)
ax2.set_ylim([0, 30])

# Note about PC regression
ax2.text(0.5, 0.97,
         '* Conv/VAE INT8 slower on PC (nn.Linear only)',
         transform=ax2.transAxes, ha='center', va='top',
         fontsize=7.5, color='#5F5E5A',
         style='italic')

fig.suptitle('Edge Device Comparison: PC (x86-64) vs Raspberry Pi 4 (AArch64)\n'
             'Mean ± std over seeds {42, 123, 456}  |  ×N = Pi4/PC slowdown factor',
             fontsize=10, fontweight='normal', y=1.02)

fig.tight_layout()
fig.savefig('outputs/figures/fig8_edge_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig('outputs/figures/fig8_edge_comparison.png', dpi=300, bbox_inches='tight')
print("Fig 8 saved.")
print("Done.")
