# Privacy-Preserving Federated Autoencoder for ECG Anomaly Detection on Edge Devices

This repository accompanies the paper *"Privacy-Preserving Federated
Autoencoder for ECG Anomaly Detection on Edge Devices"*, currently under
double-blind review at FLTA 2026.

It implements an end-to-end pipeline that combines (i) federated learning
across ten simulated hospital clients, (ii) client-side differential privacy
via DP-SGD with a Rényi-DP accountant, and (iii) INT8 post-training
quantization benchmarked on a Raspberry Pi 4. Three autoencoder families are
evaluated under a shared interface: `VanillaAE`, `ConvAE`, and `VAE`.

The mapping from paper artifacts (figures, tables) to the scripts that
produce them is given in §6 below; this is intended to make the empirical
claims in the paper independently verifiable.

---

## 1. Repository Structure

```
ecg-federated-ae/
├── configs/              # Hyperparameter dataclasses (AEConfig, VAEConfig)
├── data/                 # Preprocessed PTB-XL splits (not tracked in git)
├── evaluation/           # Metrics, ROC/PR plots, compute-cost utilities
├── fl/                   # Flower client/server, FedAvg strategy, model factory
├── models/               # VanillaAE, ConvAE, VAE, shared BaseAutoencoder
├── privacy/              # Opacus DP-SGD wrapper (Rényi-DP accountant)
├── quantisation/         # INT8 post-training quantization + Pi 4 benchmarking
├── scripts/              # Top-level entry points (preprocessing, training, eval)
├── tests/                # Smoke and VAE unit tests
├── training/             # Centralised training loops and ablations
├── utils/                # Dataset loaders, reproducibility helpers, CSV logger
├── pyproject.toml
├── requirements.txt
└── README.md
```

The code follows a strict separation between (a) **architecture-agnostic
plumbing** in `fl/`, `privacy/`, `quantisation/`, and `evaluation/`, and
(b) **architecture-specific code** in `models/`, `configs/`, and `training/`.
All three autoencoders share a single `BaseAutoencoder` interface
(`models/base.py`), which is what allows the same Flower client, the same
Opacus DP-SGD wrapper, and the same INT8 PTQ pipeline to operate on every
variant unmodified — a property the paper relies on in §III.B and §III.C.

---

## 2. Installation

Tested with Python 3.10–3.12 on Linux (CUDA 11.8 / 12.1) and macOS
(CPU / MPS). Edge benchmarking targets Raspberry Pi 4 (AArch64,
Ubuntu Server 22.04).

```bash
# Clone the (anonymized) repository
git clone <ANONYMIZED_REPO_URL>
cd ecg-federated-ae

# Recommended: isolated environment
python -m venv .venv
source .venv/bin/activate

# Install with all optional extras (FL + DP + quantization + edge)
pip install -e ".[all]"

# Or pin to the exact versions used in the paper
pip install -r requirements.txt
```

Optional extras can also be installed individually:

| Extra      | Provides                              | Required for                |
| ---------- | ------------------------------------- | --------------------------- |
| `fl`       | `flwr[simulation]`                    | Flower-based FL experiments |
| `dp`       | `opacus`                              | DP-SGD experiments          |
| `quant`    | `onnxruntime`                         | INT8 quantization pipeline  |
| `edge`     | `ptflops`                             | FLOPs / edge cost reporting |
| `dev`      | `pytest`, `pytest-cov`                | Running the test suite      |

---

## 3. Data

The pipeline operates on the publicly available **PTB-XL v1.0.3** dataset
(Wagner et al., *Scientific Data*, 2020), distributed via PhysioNet under the
Open Data Commons Attribution License v1.0. The dataset is **not redistributed
in this repository**; users must download it from PhysioNet directly.

### 3.1 Download and preprocess

```bash
# Step 1. Download PTB-XL into data/ptb-xl-raw/ (≈ 1.7 GB at 100 Hz)
#   See: https://physionet.org/content/ptb-xl/1.0.3/

# Step 2. Run the preprocessing pipeline. Applies a 4th-order zero-phase
#         Butterworth bandpass (0.05–45 Hz) per §III.A.1 of the paper, a
#         patient-level 70/15/15 split (seed 42), and per-lead z-score
#         normalization fitted on the training split only.
python scripts/preprocess_ptbxl.py \
    --raw_dir data/ptb-xl-raw \
    --output_dir data/ptb-xl-zscore \
    --sampling_rate 100 \
    --seed 42
```

The output is six `.npy` files (`{train,val,test}_{signals,labels}.npy`) with
signals shaped `(N, 12, 1000)` in `float32`. Subclass labels (MI, STTC, HYP,
CD) used for the per-class breakdown in §V.B are produced by
`scripts/extract_subclass_labels.py`.

After preprocessing, you should see the splits described in §IV.B.3 of the
paper: 14,233 / 3,078 / 3,062 patients in train / val / test, with the
training set further filtered to 6,294 normal-only records.

### 3.2 Re-normalizing pre-existing min-max scaled data

Section §IV.B.2 of the paper documents a destructive failure mode:
per-lead min-max scaling pegs each lead to its extrema, and a single
high-amplitude artefact in any one lead compresses the rest of the signal,
collapsing AUROC to ≈0.55. Replacing min-max with per-lead z-score
normalization raised centralized ConvAE AUROC to ≈0.795. If preprocessed
splits already exist under min-max scaling, `scripts/fix_normalization.py`
re-normalizes them in place without re-downloading or re-filtering PTB-XL:

```bash
python scripts/fix_normalization.py \
    --data_dir   data/ptb-xl \
    --output_dir data/ptb-xl-zscore
```

The script computes per-lead mean and standard deviation from the **training
split only** (no validation/test leakage — see §IV.B.2), applies the
resulting z-score transform to all three splits, persists the fitted
statistics as `norm_means.npy` / `norm_stds.npy` for downstream inference,
and copies any existing subclass labels and client splits across to the new
directory. New users following §3.1 can skip this step —
`preprocess_ptbxl.py` already produces z-score-normalized output directly.

### 3.3 Quick smoke-test without PTB-XL

Every training script accepts `--synthetic`, which substitutes a small
randomly generated tensor for PTB-XL. This is useful for verifying that the
pipeline runs end-to-end on a new machine before downloading the real
dataset.

---

## 4. Reproducing the Paper

The experiments are organized to mirror the paper's evaluation axes
(centralized baseline → federated learning → differential privacy → edge
quantization → component ablation). Runtime estimates assume a single
NVIDIA T4 / Colab Pro instance with 80 GB host RAM (used for FL simulation
runs; the 80 GB limit constrains `flwr` `fraction_fit`).

### 4.1 Centralised baseline (Table V, "Local")

```bash
# Three seeds (42, 123, 456) for ConvAE and VanillaAE
python scripts/train_baseline.py --model conv_ae    --data_dir data/ptb-xl-zscore --all_seeds
python scripts/train_baseline.py --model vanilla_ae --data_dir data/ptb-xl-zscore --all_seeds

# Centralised VAE (separate entry point due to KL annealing schedule and
# the ReduceLROnPlateau scheduler)
python scripts/run_vae_baseline.py --data_dir data/ptb-xl-zscore
```

### 4.2 Federated learning (Table V, "Fed."; Table VI, Configs 1–2)

The default configuration uses K=10 clients, R=50 rounds, E=5 local epochs,
a Dirichlet partition with α=0.5, and seeds {42, 123, 456}, matching §IV.B.3
and §V.A of the paper.

```bash
# Manual FedAvg path (used for the confirmed numbers in the paper)
python scripts/run_federated.py --model conv    --data_dir data/ptb-xl-zscore
python scripts/run_federated.py --model vanilla --data_dir data/ptb-xl-zscore
python scripts/run_federated.py --model vae     --data_dir data/ptb-xl-zscore

# Optional: Flower-simulation path (slower; primarily for sanity-checking).
# Note that the Flower simulation backend does not always return final
# trained weights reliably for every client under tight RAM, so the manual
# path above is the canonical one for reported results.
python scripts/run_federated.py --model conv --data_dir data/ptb-xl-zscore --use-flower
```

### 4.3 Differential privacy sweep (Table IV, Fig. 3, Fig. 4)

The ε ∈ {1, 4, 8, 24, ∞} sweep is executed inside `run_federated.py` against
the Opacus `PrivacyEngine` wrapper in `privacy/dp_sgd.py`. δ is fixed to
10⁻⁵ and the noise multiplier σ is tuned to hit each target ε after R=50
rounds, accounted via the Rényi-DP framework. The privacy unit is one ECG
recording (example-level DP), per the threat model in §VI of the paper.
Per-class breakdowns (Fig. 4) are produced by:

```bash
python scripts/run_perclass_breakdown.py --data_dir data/ptb-xl-zscore
```

### 4.4 INT8 post-training quantization (Table III, Fig. 2)

```bash
# Quantize all three architectures and write the cost CSV
python -m quantisation.ptq --model all --seed 42 --output outputs/quantisation_results.csv
```

The script measures FP32 / INT8 model size, FLOPs (via `ptflops`), and CPU
latency. Raspberry Pi 4 latencies in Table III are obtained by running the
same script directly on the device after copying the produced
`outputs/checkpoints/*.pt` files. Energy figures in the paper are computed
analytically as *E = P × t<sub>inf</sub>* with *P ≈ 4.0 W* sustained CPU
load; direct wattmeter validation is listed as future work in §VIII.

### 4.5 Component ablation (Table VI)

The seven-configuration ablation (FL × DP × INT8) is derived from the
outputs of §4.2–§4.4 and is regenerated by:

```bash
python scripts/run_evaluation.py --results outputs/fl_results.csv --compute_costs
```

### 4.6 Bottleneck ablation (paper §V, "Implementation")

Used to justify the *d=128* bottleneck choice referenced in §V of the
paper (ConvAE AUROC scales monotonically from 0.653 at d=8 to 0.771 at
d=128):

```bash
python -m training.ablation_bottleneck --model conv_ae --data_dir data/ptb-xl-zscore
```

---

## 5. Reproducibility

* The project-wide seed list is fixed in `utils/reproducibility.py`:
  `SEEDS = [42, 123, 456]`. `set_seed()` fixes Python, NumPy, and PyTorch
  (CPU + CUDA) RNGs and disables cuDNN nondeterminism.
* Exact bit-level reproducibility under Opacus and the Flower simulation
  backend is **not** guaranteed across CUDA versions; results in the paper
  are reported as mean ± std over three seeds for that reason.
* The patient-level 70/15/15 split is deterministic in PTB-XL `patient_id`
  given `--seed 42`.
* All result CSVs are written under `outputs/` and consumed by
  `scripts/run_evaluation.py` to produce the figures and tables.
* With n=3 seeds, formal nonparametric significance testing is uninformative
  (the minimum achievable Wilcoxon signed-rank p-value is 0.25); the paper
  therefore reports mean ± std and relies on effect-size magnitude, as
  discussed in §V "Statistical Testing".

---

## 6. Mapping from Paper to Code

| Paper artifact                               | Script / module                                  |
| -------------------------------------------- | ------------------------------------------------ |
| §III.A Pipeline (Fig. 1 stages)              | `scripts/preprocess_ptbxl.py` → `scripts/run_federated.py` → `quantisation/ptq.py` |
| §III.B `BaseAutoencoder` interface           | `models/base.py`, `models/{vanilla_ae,conv_ae,vae}.py` |
| §IV.A Dataset (PTB-XL one-class framing)     | `scripts/preprocess_ptbxl.py`, `scripts/extract_subclass_labels.py` |
| §IV.B.2 z-score normalization fix            | `scripts/fix_normalization.py`                   |
| §IV.B.3 Dirichlet (α=0.5) non-IID partition  | `scripts/partition_clients.py` (invoked by `run_federated.py`) |
| Fig. 2, Table III INT8 PTQ + Pi 4 latency    | `quantisation/ptq.py`                            |
| Fig. 3, Table IV DP ε-sweep                  | `scripts/run_federated.py` + `privacy/dp_sgd.py` |
| Fig. 4 Per-class DP impact                   | `scripts/run_perclass_breakdown.py`              |
| Table V Architecture comparison (local/fed)  | `scripts/train_baseline.py`, `scripts/run_federated.py` |
| Table VI Component ablation (FL × DP × INT8) | `scripts/run_evaluation.py` (consumes CSVs from above) |

---

## 7. Edge Deployment (Raspberry Pi 4)

The INT8 checkpoints produced by `quantisation/ptq.py` are PyTorch-native
and run unmodified on AArch64. The reference deployment uses:

* Raspberry Pi 4 Model B, 4 GB RAM
* Ubuntu Server 22.04 LTS (64-bit)
* Python 3.10, PyTorch 2.x (CPU build, AArch64 wheels)
* Sustained CPU governor; thermal throttling disabled during measurement

Latency figures in Table III are mean ± std over three seeds and 1 000
inference iterations per seed. Energy figures are estimated analytically
from sustained power draw; the paper flags direct wattmeter validation as
future work (§VIII).

---

## 8. Tests

```bash
pytest tests/                                       # smoke + VAE-specific unit tests
pytest tests/ --cov=. --cov-report=term-missing     # with coverage
```

---

## 9. Threat Model and Privacy Scope

The deployed mechanism is **example-level differential privacy at the
client** (local DP). The privacy unit is a single ECG recording.
Reported ε values are per-client, per-training-run budgets composed over
all R=50 rounds with δ=10⁻⁵ via the Rényi-DP accountant.

The aggregation server is honest-but-curious; secure aggregation,
Byzantine robustness, and colluding-majority reconstruction are
**out of scope** and discussed as future work. The full threat matrix
(curious server, curious client, network observer, membership-inference
attacker, gradient-inversion attacker — all mitigated up to the (ε, δ)
budget; Byzantine clients, colluding majorities, and query-time attackers
on the deployed edge model — out of scope) is given in §VI of the paper.

---

## 10. License and Citation

This source code will be released under the MIT License upon
de-anonymization (a `LICENSE` file is intentionally omitted during the
review period). The PTB-XL dataset retains its original
[ODC-BY 1.0 license](https://physionet.org/content/ptb-xl/1.0.3/).