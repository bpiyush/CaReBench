"""Plot ensemble alpha sweep results: 2-row × 6-col grid."""
import json
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Load all JSON results
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(ROOT, '_ensemble_metrics',
                        'metrics_ensemble_tarsier2-tara-cia10k-covr10k_'
                        'qwen3vlembedding-base_a*_mmebv2_all.json')
files = sorted(glob.glob(pattern))

records = []
for f in files:
    with open(f) as fh:
        records.append(json.load(fh))
records.sort(key=lambda r: r['alpha'])

alphas = np.array([r['alpha'] for r in records])

CLS_DATASETS = ['SmthSmthV2', 'HMDB51', 'UCF101', 'K700', 'Breakfast']
RET_DATASETS = ['MSR-VTT', 'MSVD', 'DiDeMo', 'YouCook2', 'VATEX']

def extract(records, task, datasets):
    out = {ds: [] for ds in datasets}
    means = []
    for r in records:
        for ds in datasets:
            out[ds].append(r['results'][task]['per_dataset'][ds])
        means.append(r['results'][task]['mean'])
    return out, means

cls_data, cls_mean = extract(records, 'cls', CLS_DATASETS)
ret_data, ret_mean = extract(records, 'ret', RET_DATASETS)

# TARA standalone (alpha = 1 reference)
TARA_CLS = {'SmthSmthV2': 76.4, 'HMDB51': 69.0, 'UCF101': 80.3,
            'K700': 59.4, 'Breakfast': 45.6, 'MEAN': 66.1}
TARA_RET = {'MSR-VTT': 40.7, 'MSVD': 82.2, 'DiDeMo': 36.8,
            'YouCook2': 16.7, 'VATEX': 53.2, 'MEAN': 45.9}

# Qwen standalone = alpha=0 record
QWEN_CLS = {ds: records[0]['results']['cls']['per_dataset'][ds] for ds in CLS_DATASETS}
QWEN_CLS['MEAN'] = records[0]['results']['cls']['mean']
QWEN_RET = {ds: records[0]['results']['ret']['per_dataset'][ds] for ds in RET_DATASETS}
QWEN_RET['MEAN'] = records[0]['results']['ret']['mean']

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.35,
    'grid.color': '#aaaaaa',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

ENSEMBLE_COLOR = '#2166ac'   # blue
TARA_COLOR     = '#d73027'   # red
QWEN_COLOR     = '#1a9641'   # green

# ---------------------------------------------------------------------------
# Build panel lists: (title, y_values, qwen_ref, tara_ref)
# ---------------------------------------------------------------------------
cls_panels = [(ds, cls_data[ds], QWEN_CLS[ds], TARA_CLS[ds]) for ds in CLS_DATASETS]
cls_panels.append(('CLS Mean', cls_mean, QWEN_CLS['MEAN'], TARA_CLS['MEAN']))
ret_panels = [(ds, ret_data[ds], QWEN_RET[ds], TARA_RET[ds]) for ds in RET_DATASETS]
ret_panels.append(('RET Mean', ret_mean, QWEN_RET['MEAN'], TARA_RET['MEAN']))

fig, axes = plt.subplots(2, 6, figsize=(18, 7), sharey=False)
fig.subplots_adjust(hspace=0.45, wspace=0.38)

def plot_panel(ax, title, y, qwen_ref, tara_ref, row_label=None):
    y = np.array(y)

    # Ensemble curve (only alpha > 0; alpha=0 is Qwen standalone)
    mask = alphas > 0
    ax.plot(alphas[mask], y[mask], color=ENSEMBLE_COLOR,
            linewidth=2.0, marker='o', markersize=4.5,
            markerfacecolor='white', markeredgewidth=1.5, zorder=3)

    # Qwen reference line (alpha = 0)
    ax.axhline(qwen_ref, color=QWEN_COLOR, linewidth=1.5,
               linestyle='--', zorder=2, label='Qwen3-VL-Emb')

    # TARA reference line
    ax.axhline(tara_ref, color=TARA_COLOR, linewidth=1.5,
               linestyle=':', zorder=2, label='TARA')

    # Mark best ensemble point
    best_idx = np.argmax(y[mask])
    best_alpha = alphas[mask][best_idx]
    best_val = y[mask][best_idx]
    ax.scatter([best_alpha], [best_val], color=ENSEMBLE_COLOR,
               s=60, zorder=4, marker='*')

    ax.set_xlim(alphas[mask].min() - 0.05, alphas[mask].max() + 0.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:g}'
    ))

    # Tighten y limits with a small pad
    all_vals = np.concatenate([y[mask], [qwen_ref, tara_ref]])
    ypad = (all_vals.max() - all_vals.min()) * 0.15 + 0.5
    ax.set_ylim(all_vals.min() - ypad, all_vals.max() + ypad)

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=5)
    ax.set_xlabel('α (TARA weight)', fontsize=8.5)
    ax.tick_params(labelsize=8)

    if row_label:
        ax.set_ylabel(row_label, fontsize=9.5, labelpad=6)

for col, (title, y, qwen_ref, tara_ref) in enumerate(cls_panels):
    row_label = 'Accuracy (%)' if col == 0 else None
    plot_panel(axes[0, col], title, y, qwen_ref, tara_ref, row_label)

for col, (title, y, qwen_ref, tara_ref) in enumerate(ret_panels):
    row_label = 'Recall@1 (%)' if col == 0 else None
    plot_panel(axes[1, col], title, y, qwen_ref, tara_ref, row_label)

# Row labels on the right
for row, label in enumerate(['Classification (CLS)', 'Retrieval (RET)']):
    axes[row, -1].annotate(
        label, xy=(1.04, 0.5), xycoords='axes fraction',
        fontsize=10, fontweight='bold', rotation=270,
        va='center', ha='left', color='#333333',
    )

# Shared legend at the top
legend_elements = [
    Line2D([0], [0], color=ENSEMBLE_COLOR, linewidth=2, marker='o',
           markersize=5, markerfacecolor='white', markeredgewidth=1.5,
           label='Ensemble (TARA + Qwen)'),
    Line2D([0], [0], color=QWEN_COLOR, linewidth=1.5, linestyle='--',
           label='Qwen3-VL-Embedding (α = 0)'),
    Line2D([0], [0], color=TARA_COLOR, linewidth=1.5, linestyle=':',
           label='TARA (standalone)'),
    Line2D([0], [0], color=ENSEMBLE_COLOR, linewidth=0, marker='*',
           markersize=9, label='Best ensemble α'),
]
fig.legend(handles=legend_elements, loc='upper center',
           ncol=4, fontsize=9.5, frameon=True,
           edgecolor='#cccccc', fancybox=False,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle(
    'Score-level Ensemble: TARA × α + Qwen3-VL-Embedding × (1 − α)\nMMEB-V2 Benchmark',
    fontsize=12.5, fontweight='bold', y=1.08,
)

save_path = os.path.join(ROOT, 'ensemble_alpha_sweep.pdf')
fig.savefig(save_path, dpi=180, bbox_inches='tight')
save_path_png = save_path.replace('.pdf', '.png')
fig.savefig(save_path_png, dpi=180, bbox_inches='tight')
print(f'Saved → {save_path}')
print(f'Saved → {save_path_png}')
plt.close()
