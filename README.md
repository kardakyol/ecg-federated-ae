# Privacy-Preserving Federated Autoencoder for ECG Anomaly Detection on Edge Devices

> **Deadline: 20 March 2026 (Friday)**  
> **Format: IEEE Access, max 10 pages + appendix**  
> **Dataset: PTB-XL (21,799 12-lead ECGs)**  
> **Stack: PyTorch + Flower (FL) + Opacus (DP) + torch.quantization (INT8)**

---

## Quick Start (EVERYONE — do this first)

```bash
git clone <repo-url>
cd ecg-federated-ae
pip install -e ".[dev]"           # installs core + pytest
pytest tests/test_smoke.py -v     # 7 tests must pass
```

If all 7 tests pass, your environment is ready. If any fail, post the error in the group chat immediately.

For specific modules:
```bash
pip install -e ".[fl]"     # Raheeb: adds Flower
pip install -e ".[dp]"     # Hilal: adds Opacus
pip install -e ".[quant]"  # Ghadah: adds ONNX Runtime + ptflops
pip install -e ".[all]"    # everything at once
```

---

## Team Roster and File Ownership

| ID | Name     | Role                          | Primary Files                                  | Sprint Focus        |
|----|----------|-------------------------------|------------------------------------------------|---------------------|
| A  | Ghouse   | Data Pipeline + Evaluation    | `data/`, `scripts/validate_data.py`            | S1: data, S2: eval  |
| B  | Shardul  | Vanilla AE + Convolutional AE | `models/vanilla_ae.py`, `models/conv_ae.py`    | S1: models, S2: FL  |
| C  | Kaan     | Variational Autoencoder       | `models/vae.py`, `configs/vae_config.py`       | S1: VAE, S2: FL     |
| D  | Raheeb   | Flower FL Pipeline            | `fl/`                                          | S1: skeleton, S2+   |
| E  | Ghadah   | Quantisation + Edge Deploy    | `quantisation/`                                | S2: prep, S3: exp   |
| F  | Sarah    | Literature + LaTeX + Diagrams | `paper/`                                       | S1: assess, S4+     |
| G  | Hilal    | DP-SGD Design + Analysis      | `privacy/`                                     | S1: assess, S2+     |
| H  | Maha     | Paper Lead + Joker Support    | `paper/`, anywhere needed                      | S1: assess, S5+     |

**Rule: You only write code in YOUR files. You only import from shared files.**

---

## Repository Structure

```
ecg-federated-ae/
│
├── .gitignore                      # data/, checkpoints/, outputs/ excluded from git
├── README.md                       # THIS FILE — read it entirely
├── pyproject.toml                  # pip install -e . uses this
├── requirements.txt                # flat dependency list
│
│   ╔══════════════════════════════════════════════════════════════════╗
│   ║  PROTECTED FILES — DO NOT MODIFY without team agreement        ║
│   ║  Everyone imports from these. Nobody writes alternatives.      ║
│   ╚══════════════════════════════════════════════════════════════════╝
│
├── models/
│   ├── __init__.py
│   ├── base.py                     # PROTECTED — BaseAutoencoder + AEOutput
│   │                               #   Raheeb calls get_parameters()/set_parameters()
│   │                               #   Ghadah calls forward() + model_size_mb()
│   │                               #   Hilal calls compute_loss() through Opacus
│   ├── vanilla_ae.py               # Shardul writes here
│   ├── conv_ae.py                  # Shardul writes here
│   └── vae.py                      # Kaan writes here
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                  # PROTECTED — compute_metrics(), aggregate_seeds()
│   │                               #   EVERYONE uses this. Nobody writes own AUROC.
│   └── plotting.py                 # PROTECTED — plot_roc(), plot_pr(), plot_bar_comparison()
│                                   #   Same colors, same DPI, same style across all figures.
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # PROTECTED — load_splits(), create_dataloaders()
│   │                               #   Supports .npy and .pt formats from Ghouse.
│   │                               #   Auto-fixes (N,1000,12) → (N,12,1000).
│   │                               #   Train loader = normal samples only.
│   ├── csv_logger.py               # PROTECTED — ResultLogger with STANDARD_COLUMNS
│   │                               #   Everyone logs experiment results through this.
│   └── reproducibility.py          # PROTECTED — SEEDS=[42,123,456], set_seed(), get_device()
│
│   ╔══════════════════════════════════════════════════════════════════╗
│   ║  PERSONAL FILES — each person writes in their own directory    ║
│   ╚══════════════════════════════════════════════════════════════════╝
│
├── configs/                        # Kaan: vae_config.py, Shardul: ae_config.py
│   └── __init__.py
│
├── training/                       # Shardul + Kaan: centralised training loops
│   └── __init__.py
│
├── fl/                             # Raheeb: Flower server + client
│   ├── __init__.py
│   └── flower_client.py            # Raheeb writes here
│
├── quantisation/                   # Ghadah: PTQ pipeline + edge benchmarks
│   ├── __init__.py
│   └── ptq.py                      # Ghadah writes here
│
├── privacy/                        # Hilal: Opacus DP-SGD integration
│   ├── __init__.py
│   └── dp_sgd.py                   # Hilal writes here
│
├── scripts/
│   ├── __init__.py
│   └── validate_data.py            # Ghouse runs after preprocessing
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py               # Everyone runs this to verify setup
│
├── paper/                          # Sarah + Maha: LaTeX files, diagrams
│
├── data/                           # NOT IN GIT — Ghouse outputs go here
│   └── ptb-xl/                     #   train_signals.npy, train_labels.npy, etc.
│
├── checkpoints/                    # NOT IN GIT — model weights
└── outputs/                        # NOT IN GIT — CSVs, figures, logs
    └── figures/
```

---

## Data Setup (EVERYONE — do this to get the dataset)

### Step 1: Download raw PTB-XL

Download the PTB-XL dataset from Kaggle:
**https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3?resource=download**

1. Go to the link above and click **Download**
2. Extract the downloaded zip
3. Place the extracted contents into `data/ptb-xl-raw/` so the structure looks like:

```
data/ptb-xl-raw/
  ptbxl_database.csv
  scp_statements.csv
  records100/
  records500/
  ...
```

### Step 2: Run preprocessing

```bash
python scripts/preprocess_ptbxl.py --raw_dir data/ptb-xl-raw --output_dir data/ptb-xl --seed 42
```

This applies bandpass filtering (0.05-45 Hz), baseline wander removal, min-max normalisation per lead, binary label encoding (NORM=0, abnormal=1), and patient-level train/val/test split (70/15/15).

### Step 3: Run non-IID client partitioning

```bash
python scripts/partition_clients.py --data_dir data/ptb-xl --alpha 0.5 --num_clients 10 --seed 42
```

This creates Dirichlet-distributed client splits for federated learning and saves a per-client distribution histogram to `outputs/figures/client_distribution.pdf`.

### Step 4: Validate

```bash
python scripts/validate_data.py --data_dir data/ptb-xl
```

You should see:
```
  OK train: 14233 samples (6294 normal, 7939 abnormal)
  OK val: 3078 samples (1337 normal, 1741 abnormal)
  OK test: 3062 samples (1407 normal, 1655 abnormal)

All checks passed!
```

## Step 5: Fix Normalization (CRITICAL)

To ensure more stable training and faster convergence of AE/VAE models, you must convert the initial Min-max normalization (0,1) to Z-score normalization (μ=0,σ=1):

```bash
python fix_normalization.py --data_dir data/ptb-xl --output_dir data/ptb-xl-zscore
```
Note: From this step onwards, use data/ptb-xl-zscore as your default data directory for all training and evaluation scripts.

### Step 6: Generate per-class subclass labels (required for per-class breakdown)

The per-class evaluation (MI, STTC, HYP, CD breakdown) requires mapping each test sample back to its PTB-XL diagnostic superclass. This step reads the raw PTB-XL metadata and produces `.npy` label files.

**Prerequisites:** You need the raw PTB-XL metadata files (`ptbxl_database.csv` and `scp_statements.csv`) from Step 1.

```bash
python scripts/extract_subclass_labels.py --raw_dir data/ptb-xl-raw --data_dir data/ptb-xl
```

**On Google Colab**, if you don't have the raw metadata locally, download it first:

```bash
wget -q https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv -P data/ptb-xl-raw/
wget -q https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv -P data/ptb-xl-raw/

python scripts/extract_subclass_labels.py --raw_dir data/ptb-xl-raw --data_dir data/ptb-xl
```

This produces:
```
data/ptb-xl/test_subclass_labels.npy    # (N_test,) int: {-1,0,1,2,3,4}
data/ptb-xl/val_subclass_labels.npy     # (N_val,)
data/ptb-xl/train_subclass_labels.npy   # (N_train,)
data/ptb-xl/subclass_map.json           # {"NORM":0, "MI":1, "STTC":2, "HYP":3, "CD":4, "UNKNOWN":-1}
```

Verify with:
```bash
python scripts/extract_subclass_labels.py --data_dir data/ptb-xl --check
```

**Note:** If you are using z-score normalised data, point `--data_dir` to `data/ptb-xl-zscore` instead.

---
## Sprint 2: Federated AE Updates (Raheeb)

### How to test/reproduce the Federated Setup:
Run the following commands to verify the pipeline across all models and $\alpha$ (alpha) settings:

| Model | $\alpha=0.1$ | $\alpha=0.5$ | $\alpha=1.0$ |
| :--- | :--- | :--- | :--- |
| **Vanilla** | `python fl/flower_server.py --model vanilla --alpha 0.1 --epochs 10 --rounds 20` | `python fl/flower_server.py --model vanilla --alpha 0.5 --epochs 10 --rounds 20` | `python fl/flower_server.py --model vanilla --alpha 1.0 --epochs 10 --rounds 20` |
| **Conv** | `python fl/flower_server.py --model conv --alpha 0.1 --epochs 10 --rounds 20` | `python fl/flower_server.py --model conv --alpha 0.5 --epochs 10 --rounds 20` | `python fl/flower_server.py --model conv --alpha 1.0 --epochs 10 --rounds 20` |
| **VAE** | `python fl/flower_server.py --model vae --alpha 0.1 --epochs 10 --rounds 20` | `python fl/flower_server.py --model vae --alpha 0.5 --epochs 10 --rounds 20` | `python fl/flower_server.py --model vae --alpha 1.0 --epochs 10 --rounds 20` |

* **Plots:** outputs/figures/ for the convergence PNGs.
* **CSV:** outputs/convergence_results.csv for the populated metrics
---
## Sprint 3 Full 6 Config Ablation Table (Raheeb)

### How to test/reproduce the test:

| #  | FL | DP         | Quantisation | Commands to run |
|----|----|-----------|-------------|------------------------|
| 2  | ✓  | ✗         | ✗           | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 'inf' --precision_type "fp32" --seed [42/123/456]` |
| 3  | ✓  | ✓ (ε=8)  | ✗           | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 8 --precision_type "fp32" --seed [42/123/456]` |
| 4  | ✓  | ✗         | ✓ (INT8)    | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 'inf' --precision_type "int8" --seed [42/123/456]` |
| 5  | ✓  | ✓ (ε=4)  | ✓ (INT8)    | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 4 --precision_type "int8" --seed [42/123/456]` |
| 6  | ✓  | ✓ (ε=1)  | ✗           | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 1 --precision_type "fp32" --seed [42/123/456]` |
| 7  | ✓  | ✓ (ε=1)  | ✓ (INT8)    | `python fl/flower_server.py --model [vanilla/conv/vae] --alpha [0.1/0.5/1.0] --epochs 5 --rounds 50 --clients 10 --epsilon 1 --precision_type "int8" --seed [42/123/456]` |

* **Ablation Result:** Saved in the directory `outputs/ablation_results.csv`
---

## Data Contract (Ghouse → Everyone)

Preprocessed PTB-XL data lives in `data/ptb-xl/`. Everyone else loads it via:

```python
from utils.dataset import load_splits, create_dataloaders

splits = load_splits("data/ptb-xl")
loaders = create_dataloaders(splits, batch_size=128)
# loaders["train"]      → normal samples only (unsupervised AD)
# loaders["val"]        → all samples (for monitoring)
# loaders["val_normal"] → normal val samples (for threshold calibration)
# loaders["test"]       → all samples (for final evaluation)
```

### File format:

| File                    | dtype      | Shape           | Notes                            |
|-------------------------|------------|-----------------|----------------------------------|
| `train_signals.npy`    | float32    | (N, 12, 1000)   | Channels-FIRST (12 leads, 1000 timesteps) |
| `train_labels.npy`     | int64      | (N,)            | 0 = normal, 1 = abnormal        |
| `val_signals.npy`      | float32    | (N, 12, 1000)   | Patient-level split, no leakage  |
| `val_labels.npy`       | int64      | (N,)            |                                  |
| `test_signals.npy`     | float32    | (N, 12, 1000)   |                                  |
| `test_labels.npy`      | int64      | (N,)            |                                  |

### Before the real data is ready, you can test with synthetic data:
```python
from utils.dataset import create_synthetic_data, create_dataloaders

splits = create_synthetic_data(n_train=2000, n_val=500, n_test=500)
loaders = create_dataloaders(splits, batch_size=128)
```

---

## Model Contract (Shardul + Kaan → Raheeb, Ghadah, Hilal)

ALL autoencoder models MUST extend `BaseAutoencoder` from `models/base.py`.

### Shardul — what you must do:

```python
# In models/vanilla_ae.py and models/conv_ae.py:

from models.base import BaseAutoencoder, AEOutput
import torch.nn.functional as F

class VanillaAE(BaseAutoencoder):
    def __init__(self):
        super().__init__()
        # your layers here — use GroupNorm, NOT BatchNorm
        # use inplace=False in activations

    def forward(self, x):
        # x: (batch, 12, 1000)
        # ... your encoder + decoder ...
        return AEOutput(x_hat=reconstructed)   # x_hat.shape MUST equal x.shape

    def compute_loss(self, x, output, **kwargs):
        mse = F.mse_loss(output.x_hat, x)
        return (mse,)                          # tuple, first element = total loss
```

### Kaan — what you must do:

```python
# In models/vae.py:

from models.base import BaseAutoencoder, AEOutput

class VAE(BaseAutoencoder):
    def forward(self, x):
        # ... encoder → mu, logvar → reparameterise → decoder ...
        return AEOutput(x_hat=reconstructed, mu=mu, logvar=logvar, z=z)

    def compute_loss(self, x, output, beta=1.0, **kwargs):
        mse = F.mse_loss(output.x_hat, x)
        kl = -0.5 * torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp())
        kl = kl / x.shape[0]  # mean over batch
        total = mse + beta * kl
        return (total, mse, kl)   # first = total for backward, rest = logging
```

### CRITICAL rules for both Shardul and Kaan:
- **NO BatchNorm** — Hilal's Opacus cannot compute per-sample gradients through it. Use `nn.GroupNorm(num_groups, channels)` or `nn.InstanceNorm1d(channels)`.
- **NO inplace=True** — `nn.LeakyReLU(inplace=False)`, `nn.ReLU(inplace=False)`. Opacus needs these.
- **x_hat.shape == x.shape** — if ConvTranspose produces 1001 instead of 1000, fix with `F.interpolate(x_hat, size=x.shape[-1])`.
- **Do NOT override** `get_parameters()` or `set_parameters()` — they are inherited and Raheeb's Flower client depends on them.

### Why this matters — downstream consumers:

```python
# Raheeb's Flower client does this for ANY model:
params = model.get_parameters()           # works for VanillaAE, ConvAE, VAE
model.set_parameters(aggregated_params)   # FedAvg result from server

# Ghadah's quantisation does this for ANY model:
output = model(x)                         # .x_hat guaranteed
size = model.model_size_mb()              # FP32 size

# Hilal's Opacus does this for ANY model:
loss, *_ = model.compute_loss(x, output)  # first element = total loss
loss.backward()                           # Opacus hooks into gradients
```

---

## Metric Contract (EVERYONE)

### Computing metrics — ONE function, no alternatives:
```python
from evaluation.metrics import compute_metrics, aggregate_seeds

result = compute_metrics(y_true, anomaly_scores, threshold)
print(result.auroc, result.auprc, result.sensitivity, result.specificity, result.f1)
```

### Aggregating over 3 seeds:
```python
from utils.reproducibility import SEEDS
results_per_seed = []
for seed in SEEDS:  # [42, 123, 456]
    # ... train and evaluate ...
    results_per_seed.append(result)
agg = aggregate_seeds(results_per_seed)
# agg["auroc"]["mean"], agg["auroc"]["std"]
```

### Logging results to CSV:
```python
from utils.csv_logger import ResultLogger
logger = ResultLogger("outputs/my_experiment.csv")
logger.log(
    model="vae", setting="federated", beta=0.5, epsilon=4.0,
    precision_type="fp32", seed=42, auroc=0.92, auprc=0.88,
    sensitivity=0.85, specificity=0.95, precision_score=0.90, f1=0.87,
    model_size_mb=1.2, inference_latency_ms=3.5, training_time_s=120.0,
)
```

### Plotting — shared functions, consistent style:
```python
from evaluation.plotting import plot_roc, plot_pr, plot_bar_comparison

plot_roc({"VanillaAE": result_vanilla, "VAE": result_vae}, save_path="outputs/figures/roc.pdf")
plot_pr({"VanillaAE": result_vanilla, "VAE": result_vae}, save_path="outputs/figures/pr.pdf")
```

---

## Sprint-by-Sprint Breakdown

### Sprint 1: Data Pipeline + Baseline Autoencoders (Mar 2–5)

**Checkpoint: 5 Mar evening — 3 AE models produce centralised baselines. FL skeleton runs with dummy model.**

| Person   | What to do | Where to put it | How to verify |
|----------|-----------|-----------------|---------------|
| Ghouse   | Download PTB-XL, preprocess (bandpass 0.05–100Hz, baseline wander removal, min-max normalise per lead), patient-level train/val/test split (70/15/15), non-IID Dirichlet partitioning (α=0.5, K=10) | `data/ptb-xl/*.npy` and `data/ptb-xl/client_splits/` | `python scripts/validate_data.py --data_dir data/ptb-xl` — all checks pass |
| Shardul  | Implement VanillaAE (FC, bottleneck=32) and ConvAE (Conv1d). Must extend `BaseAutoencoder`. Centralised training on normal-only, evaluate with `compute_metrics()`. Record AUROC, AUPRC, Sens, Spec, F1 | `models/vanilla_ae.py`, `models/conv_ae.py`, `training/train_ae.py` | Import works: `from models.vanilla_ae import VanillaAE` and model passes DummyAE-style test in smoke test |
| Kaan     | Implement β-VAE (Conv1d encoder/decoder, latent_dim=32, β=0.1/0.5/1.0). Must extend `BaseAutoencoder`. Centralised training, anomaly scoring (MSE + KL), evaluate with `compute_metrics()` | `models/vae.py`, `configs/vae_config.py`, `training/train_vae.py` | Import works: `from models.vae import VAE` and β sweep produces 3 result rows in CSV |
| Raheeb   | Install Flower, build server config (FedAvg), create client class template with fit()/evaluate(), test end-to-end with a dummy model | `fl/flower_client.py`, `fl/flower_server.py` | `python fl/flower_server.py --dry-run` completes without error |
| Ghadah   | (Assessment 1) Design computation efficiency measurement framework, define metrics (size/FLOPs/latency/memory/power), compile Assessment 1 document | `paper/assessment1/` | Assessment 1 PDF ready by 6 Mar morning |
| Maha    | (Assessment 1) Write literature review (500–700 words, 3 thematic subsections), include gap matrix table | `paper/assessment1/`, `paper/references.bib` | Literature review section complete in LaTeX |
| Hilal    | (Assessment 1) Write proposal section (research problem + 4 research questions RQ1–RQ4), justify significance | `paper/assessment1/` | Proposal section complete, 4 RQs clearly stated |
| Sarah     | (Assessment 1) System architecture diagram, component diagram, UML activity diagram, support blockers | `paper/assessment1/`, `paper/diagrams/` | Diagrams exportable as PDF/PNG, Assessment 1 draft compiled |

---

### Sprint 2: Federated Training + First FL Results (Mar 6–8)

**Assessment 1 submitted 6 Mar morning. Full implementation focus from here.**

**Checkpoint: 8 Mar evening — FL pipeline produces federated results for all 3 AEs. Quantisation and DP modules ready for integration.**

| Person   | What to do | Where to put it | How to verify |
|----------|-----------|-----------------|---------------|
| Shardul + Kaan | Integrate all 3 AEs into Flower clients, federated training (R=50 rounds, E=5 local epochs), produce centralised vs. federated comparison table, 3 random seeds | `fl/` (work with Raheeb), results in `outputs/fl_results.csv` | CSV has 9 rows (3 models × 3 seeds) with federated AUROC |
| Raheeb   | Client selection strategy, post-global fine-tuning (10 local epochs), non-IID experiments (Dirichlet α = 0.1, 0.5, 1.0), training convergence curves | `fl/flower_client.py`, `fl/flower_server.py`, `fl/strategies.py` | Non-IID sweep produces 3×3 result rows, convergence CSVs exist |
| Ghadah   | Learn PyTorch quantisation API, prepare PTQ script (FP32→INT8), test quantised inference on all 3 AEs (centralised), measure FP32 vs INT8 model size, set up Raspberry Pi 4 | `quantisation/ptq.py`, `quantisation/benchmarks.py` | `python quantisation/ptq.py --model vanilla_ae` produces size comparison |
| Ghouse   | Build shared evaluation automation: ROC/PR curve plotting, results logging to CSV, computation metrics script (FLOPs via ptflops, inference time, peak memory), statistical significance tests (Wilcoxon) | `evaluation/` (enhance existing), `scripts/run_evaluation.py` | `python scripts/run_evaluation.py --results outputs/fl_results.csv` generates figures |
| Hilal    | Install Opacus, implement DP-SGD (per-sample gradient clipping + Gaussian noise), privacy accountant for ε tracking (Rényi DP), refine literature review | `privacy/dp_sgd.py`, `privacy/accountant.py` | `from privacy.dp_sgd import make_private` importable without error |
| Sarah    | Set up IEEE Access LaTeX template, section stubs, begin system architecture diagram (draw.io/TikZ), collect additional references | `paper/main.tex`, `paper/figures/`, `paper/references.bib` | LaTeX compiles to PDF with correct IEEE formatting |
| Maha     | Help Hilal with Opacus + Flower integration, begin Methodology section skeleton, coordinate sprint progress, prepare LaTeX table/figure templates | `paper/`, help in `privacy/` | Methodology skeleton has subsection headers in LaTeX |

---

### Sprint 3: Quantisation + DP Experiments (Mar 9–12)

**This is the core experimental sprint. Both dimensions measured here. NO MORE EXPERIMENTS AFTER 12 MAR.**

**Checkpoint: 12 Mar evening — ALL experiment tables complete with mean ± std. Edge device benchmarks complete.**

| Person   | What to do | Where to put it |
|----------|-----------|-----------------|
| Ghadah   | Run 3 AE × {FP32, INT8} = 6 federated configs. Measure per config: model size (MB), FLOPs, inference latency (ms), peak memory (MB). Raspberry Pi 4 inference benchmarks. Produce quantisation accuracy degradation table. | `quantisation/`, results in `outputs/quantisation_results.csv` |
| Hilal    | Design DP sweep on best AE: ε = {1, 2, 4, 8, ∞}. Analyse per ε: AUROC, AUPRC, Sens, Spec, training time overhead. Produce privacy–utility trade-off curve. Write DP results narrative. | `privacy/`, results in `outputs/dp_sweep_results.csv` |
| Raheeb   | Execute DP sweep experiments (Opacus + Flower), run combined DP + quantisation configs (at least 2), measure DP overhead on training time, per-class DP impact analysis | `fl/`, results in `outputs/dp_fl_results.csv` |
| Shardul + Kaan | Bottleneck size ablation {16, 32, 64, 128} for best AE, layer depth ablation (shallow vs deep), per-class anomaly detection breakdown (MI, STTC, HYP, CD) | `training/`, results in `outputs/ablation_architecture.csv` |
| Ghouse + Raheeb | Run 7-configuration component ablation (see table below), 3 seeds per config, Wilcoxon signed-rank test between key pairs | Results in `outputs/ablation_results.csv` |
| Maha     | Support Ghadah with Pi4 benchmarks, help with ablation execution, begin drafting methodology section | `paper/` |

#### Component Ablation Table (7 configs):

| #  | FL | DP         | Quantisation | Expected Output Columns |
|----|----|-----------|-------------|------------------------|
| 1  | ✗  | ✗         | ✗           | auroc, size_mb, latency_ms (centralised baseline) |
| 2  | ✓  | ✗         | ✗           | + training_time_per_round_s |
| 3  | ✓  | ✓ (ε=4)  | ✗           | + epsilon column |
| 4  | ✓  | ✗         | ✓ (INT8)    | + precision_type=int8 |
| 5  | ✓  | ✓ (ε=4)  | ✓ (INT8)    | combined |
| 6  | ✓  | ✓ (ε=1)  | ✗           | strict privacy |
| 7  | ✓  | ✓ (ε=1)  | ✓ (INT8)    | strict privacy + quantised |

---

### Sprint 4: Final Experiments + Figures + LaTeX (Mar 13–15)

**Checkpoint: 15 Mar evening — All figures, tables, LaTeX skeleton READY. Experiment phase OVER.**

| Person   | What to do | Where to put it |
|----------|-----------|-----------------|
| Ghouse + Raheeb | Client scalability test (K=5,10,20), re-run inconsistent results, compile final summary CSV with ALL results | `outputs/final_results.csv` |
| Ghadah   | Pareto front figure (AUROC vs FLOPs/model size), ROC overlay (3 AEs × FP32+INT8), quantisation impact bar chart, edge device comparison (GPU vs Pi4), all 300dpi PDF | `outputs/figures/` |
| Hilal    | DP trade-off curve (ε vs AUROC), DP training overhead figure, per-class DP impact visualisation, draft Results privacy subsection | `outputs/figures/`, `paper/` |
| Raheeb   | Training convergence curves (round vs loss per AE), non-IID severity (α) vs AUROC bar chart | `outputs/figures/` |
| Sarah    | IEEE LaTeX template configured, all table shells created, system architecture diagram complete, AE architecture diagrams (3 variants), reference list (12 core + ~15 additional) | `paper/` |
| Shardul + Kaan | Hyperparameter tables for appendix, per-client performance breakdown, additional ablation variants, MIT-BIH generalizability test (appendix only, if time) | `paper/appendix/`, `outputs/` |
| Maha     | LaTeX section headings + placeholder structure, empty table templates in IEEE format, support anyone behind schedule, begin writing Methodology draft | `paper/` |

---

### Sprint 5: Paper Writing — FULL FOCUS (Mar 16–18)

**Checkpoint: 18 Mar evening — Complete first draft compiled as single PDF.**

#### 16 March (Monday) — Introduction + Methodology

| Person  | Section | Length | Content |
|---------|---------|--------|---------|
| Maha    | Abstract + I. Introduction | 0.3 + 1.2 pages | Problem, motivation, 4–5 contributions, paper outline |
| Sarah   | II. Related Work | 1.0 page | Condense 12-paper review, gap matrix in appendix |
| Raheeb  | III. Methodology (FL part) | 1.0 page | FL pipeline, AE architectures, FedAvg, non-IID |
| Hilal   | III. Methodology (DP + quant part) | 1.0 page | DP-SGD formulation, quantisation strategy |

#### 17 March (Tuesday) — Experiments + Results

| Person  | Section | Length | Content |
|---------|---------|--------|---------|
| Shardul + Kaan | IV. Experimental Setup | 1.0 page | Dataset, preprocessing, implementation details, hyperparameters, metrics |
| Ghadah  | V. Results (Computation) | ~1.2 pages | Quantisation impact, FLOPs, latency, Pareto front, edge benchmarks |
| Hilal   | V. Results (Privacy) | ~1.2 pages | DP epsilon sweep, privacy-utility curve, per-class DP impact |
| Ghouse  | V. Results (Ablation) | ~1.1 pages | Architecture comparison, component ablation, scalability |

#### 18 March (Wednesday) — Integration + Polish

| Person  | Task |
|---------|------|
| Maha    | Write VI. Conclusion + Future Work (0.5 page). Merge ALL sections into single coherent voice. Verify ≤10 pages, move excess to appendix. Complete all cross-references. |
| Hilal   | Proofread Privacy + Methodology. Ensure DP formulation is mathematically consistent. |
| Everyone | Proofread your own section. Flag inconsistencies. |

---

### Sprint 6: Review + Submission (Mar 19–20)

#### 19 March (Thursday)
- Everyone reads the complete paper end-to-end and flags errors
- Cross-reference check: table/figure numbers, citation numbers
- Grammar check (Grammarly or similar)
- IEEE format compliance: margins, fonts, section numbering
- Prepare appendix: supplementary tables, hyperparameters, extra figures
- Clean GitHub repository, verify README is accurate
- Verify contribution table is accurate and agreed by all

#### 20 March (Friday) — SUBMISSION DAY
- Maha: final read-through (single person, fresh eyes)
- Compile final PDF from LaTeX
- **SUBMIT: PDF (report) + ZIP (software artifact) on NESS**

---

## Required Figures and Tables for Paper

| #     | Type      | Content                                               | Owner           |
|-------|-----------|-------------------------------------------------------|-----------------|
| Fig 1 | Diagram   | System architecture: FL + AE + DP + Quantisation      | Sarah + Maha    |
| Fig 2 | Diagram   | AE architecture diagrams (3 variants side by side)     | Shardul + Kaan  |
| Fig 3 | Plot      | ROC curves: 3 AEs × FP32 + INT8 overlay               | Ghadah          |
| Fig 4 | Plot      | Computation Pareto front: AUROC vs model size/FLOPs    | Ghadah          |
| Fig 5 | Plot      | DP epsilon vs AUROC trade-off curve                    | Hilal           |
| Fig 6 | Plot      | Training convergence curves (round vs loss per AE)     | Raheeb          |
| Fig 7 | Bar chart | Non-IID severity (α) vs AUROC                         | Raheeb          |
| Fig 8 | Bar chart | Edge device comparison (GPU vs Pi4 latency)            | Ghadah          |
| Tab I | Table     | AE architecture comparison (AUROC, AUPRC, sens, spec, F1) | Shardul + Kaan |
| Tab II| Table     | Computation efficiency (size, FLOPs, latency, memory, Pi4) | Ghadah        |
| Tab III| Table    | DP epsilon sweep results                               | Hilal           |
| Tab IV| Table     | Component ablation study (7 configs)                   | Ghouse + Raheeb |

All plots: use `from evaluation.plotting import plot_roc, plot_pr, plot_bar_comparison, COLORS`. Export as 300 dpi PDF for LaTeX.

---

## Protected Files — What They Do and Why

### `models/base.py` — The Interface Contract

Defines `BaseAutoencoder` (abstract class) and `AEOutput` (dataclass). Every model extends `BaseAutoencoder` and returns `AEOutput` from `forward()`. This guarantees that Raheeb, Ghadah, and Hilal can write code that works with ANY model without knowing its internals.

**If you change this file, you break Raheeb's Flower client, Ghadah's quantisation pipeline, and Hilal's DP-SGD wrapper simultaneously.**

### `evaluation/metrics.py` — Single Source of Truth for Metrics

One function: `compute_metrics(y_true, scores, threshold)` returns `MetricsResult` with AUROC, AUPRC, sensitivity, specificity, precision, F1, ROC/PR curves. `aggregate_seeds()` computes mean ± std over 3 seeds.

**If you write your own AUROC function, your numbers will differ from everyone else's and the paper will have inconsistent results.**

### `evaluation/plotting.py` — Consistent Visual Style

Shared colors, fonts, DPI settings. `plot_roc()`, `plot_pr()`, `plot_bar_comparison()` produce publication-quality 300 dpi figures in IEEE-compatible style.

**If you use your own matplotlib style, figures in the paper will look inconsistent.**

### `utils/dataset.py` — Unified Data Loading

`load_splits()` loads Ghouse's preprocessed data. `create_dataloaders()` creates train (normal-only), val, val_normal (for threshold calibration), and test loaders. `create_synthetic_data()` generates fake ECG-like data for pipeline testing.

**If you write your own data loader, you risk loading channels-last instead of channels-first, or including abnormal samples in training.**

### `utils/csv_logger.py` — Standardised Output Format

`ResultLogger` with `STANDARD_COLUMNS` ensures every experiment result CSV has the same columns. This makes final result aggregation for the paper trivial.

### `utils/reproducibility.py` — Seeds and Determinism

`SEEDS = [42, 123, 456]`. `set_seed()` fixes Python, NumPy, PyTorch, and CUDA random states. `get_device()` handles GPU/MPS/CPU fallback.

**Always use these seeds. The paper reports mean ± std over these 3 seeds.**

---

## Opacus Compatibility Rules (Shardul + Kaan MUST follow)

Hilal will wrap your models with Opacus for DP-SGD in Sprint 3. Opacus computes per-sample gradients, which is incompatible with certain layer types:

| DO NOT use             | USE instead                              |
|-----------------------|------------------------------------------|
| `nn.BatchNorm1d(C)`  | `nn.GroupNorm(num_groups=min(32,C), num_channels=C)` |
| `nn.BatchNorm2d(C)`  | `nn.GroupNorm(num_groups=min(32,C), num_channels=C)` |
| `inplace=True`        | `inplace=False` in ALL activations       |

If you use BatchNorm, Hilal will have to rewrite your model in Sprint 3, losing valuable experiment time.

---

## Git Workflow

### Branch naming:
```
feature/ghouse-data-pipeline
feature/shardul-vanilla-ae
feature/kaan-vae
feature/raheeb-flower-client
feature/ghadah-quantisation
feature/hilal-dp-sgd
```

### Commit messages:
```
[Ghouse] Add PTB-XL preprocessing pipeline
[Shardul] Implement VanillaAE extending BaseAutoencoder
[Kaan] Add beta-VAE with KL annealing
[Raheeb] FL pipeline end-to-end with FedAvg
```

### Before pushing:
```bash
pytest tests/test_smoke.py -v   # must pass
```

### Never commit:
- `data/` (large .npy files)
- `checkpoints/` (model weights)
- `outputs/` (experiment results — each person generates locally)
- `.gitignore` already handles these

---

## 10-Page Budget Plan

| Section                | Pages | Writer            |
|------------------------|-------|-------------------|
| Abstract               | 0.3   | Maha              |
| I. Introduction        | 1.2   | Maha              |
| II. Related Work       | 1.0   | Sarah             |
| III. Methodology       | 2.0   | Raheeb + Hilal    |
| IV. Experimental Setup | 1.0   | Shardul + Kaan    |
| V. Results & Discussion| 3.5   | Ghadah + Hilal + Ghouse |
| VI. Conclusion         | 0.5   | Maha              |
| References             | 0.5   | Sarah             |
| **TOTAL**              | **10.0** |                |

Appendix (no page limit): full gap matrix, hyperparameter tables, per-client breakdown, additional ROC/PR curves, bottleneck/depth ablation, edge device benchmarks, MIT-BIH generalizability results, code repository link.

---

## Daily Standup Protocol

**Every day at 22:00 — 15 minutes Discord/Teams:**
- What did you complete today? (30 sec per person)
- What will you do tomorrow? (30 sec per person)
- Any blockers? (30 sec per person)

---

## Critical Success Factors

1. **Complete ablation study** — The 7-config table with mean±std is non-negotiable
2. **Two clear dimension narratives** — Privacy (ε sweep curve) and Computation (Pareto front) with dedicated figures and tables
3. **Concrete numbers** — Every claim backed by metrics: "X MB", "Y ms", "Z AUROC at ε=4"
4. **DP epsilon sweep with ≥5 values** — Shows the full privacy–utility curve
5. **Quantisation before/after** — FP32 vs INT8 for all 3 AEs, both accuracy and compute
6. **Edge device benchmarks** — Real Pi4 or ONNX Runtime simulation
7. **Statistical rigour** — 3 seeds, mean±std, Wilcoxon signed-rank for key comparisons
8. **Reproducibility** — Fixed seeds, open-source code, documented hyperparameters
