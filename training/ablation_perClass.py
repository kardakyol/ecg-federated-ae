import argparse
import ast
import os
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from utils.dataset import load_splits, create_dataloaders
from utils.reproducibility import SEEDS, set_seed, get_device, setup_logging
from utils.csv_logger import ResultLogger
from evaluation.metrics import compute_metrics, aggregate_seeds, format_aggregated

from models.vanilla_ae import VanillaAE
from models.conv_ae import ConvAE


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "vanilla_ae": VanillaAE,
    "conv_ae": ConvAE,
}


TARGET_CLASSES = ["MI", "STTC", "HYP", "CD"]


def build_perclass_masks(raw_dir, data_dir):
    """Map test samples to superclasses using PTB-XL metadata."""
    metadata_path = os.path.join(raw_dir, "ptbxl_database.csv")
    scp_path = os.path.join(raw_dir, "scp_statements.csv")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Cannot find {metadata_path}\n"
            f"Need original PTB-XL metadata for per-class breakdown.\n"
            f"Download from: https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3"
        )

    metadata = pd.read_csv(metadata_path, index_col="ecg_id")
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def get_superclasses(scp_dict):
        """Map SCP codes to diagnostic superclasses."""
        result = set()
        for key, confidence in scp_dict.items():
            if key in scp_df.index and confidence > 0:
                result.add(scp_df.loc[key].diagnostic_class)
        return list(result)

    metadata["diagnostic_superclass"] = metadata.scp_codes.apply(get_superclasses)

    metadata["is_normal"] = metadata.diagnostic_superclass.apply(
        lambda x: x == ["NORM"]
    )

    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    n_test = len(test_labels)

    fold_map_path = os.path.join(data_dir, "test_ecg_ids.npy")
    if os.path.exists(fold_map_path):
        test_ecg_ids = np.load(fold_map_path)
        logger.info(f"Loaded test ECG IDs from {fold_map_path}")
    else:
        logger.warning("test_ecg_ids.npy not found. Using strat_fold=10 as test set.")
        test_ecg_ids = metadata[metadata.strat_fold == 10].index.values

        if len(test_ecg_ids) != n_test:
            logger.warning(
                f"Fold 10 has {len(test_ecg_ids)} records but test set has {n_test}. "
                f"Counts don't match — trying folds 9+10..."
            )
            test_ecg_ids = test_ecg_ids[:n_test]

    normal_mask = test_labels == 0

    masks = {}
    for cls in TARGET_CLASSES:
        cls_mask = np.zeros(n_test, dtype=bool)

        for i, ecg_id in enumerate(test_ecg_ids[:n_test]):
            if ecg_id in metadata.index:
                superclasses = metadata.loc[ecg_id, "diagnostic_superclass"]
                if cls in superclasses:
                    cls_mask[i] = True

        n_cls = cls_mask.sum()
        masks[cls] = cls_mask
        logger.info(f"  {cls}: {n_cls} abnormal samples in test set")

    logger.info(f"  NORM: {normal_mask.sum()} normal samples in test set")

    return masks, normal_mask



def compute_anomaly_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            output = model(signals)
            mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def compute_anomaly_scores_raw(model, signals_tensor, device, batch_size=128):
    """Compute scores for a raw tensor (not a DataLoader)."""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(signals_tensor), batch_size):
            batch = signals_tensor[i:i+batch_size].to(device)
            output = model(batch)
            mse = ((output.x_hat - batch) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.cpu().numpy())
    return np.concatenate(all_scores)


def find_threshold(model, val_normal_loader, device, percentile=95):
    model.eval()
    scores = []
    with torch.no_grad():
        for signals, labels in val_normal_loader:
            signals = signals.to(device)
            output = model(signals)
            mse = ((output.x_hat - signals) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())
    return float(np.percentile(np.concatenate(scores), percentile))



# Training 
def train_and_evaluate_perclass(model_name, loaders, seed, device, test_signals,
                                 test_labels, class_masks, normal_mask, threshold_from_val,
                                 epochs=50, lr=1e-3, weight_decay=1e-5, bottleneck=32):
    """Train one model, evaluate per-class."""
    set_seed(seed)
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(bottleneck=bottleneck).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    logger.info(f"  {model_name} | seed={seed} | params={model.count_parameters():,}")

    best_val_auroc = 0.0
    best_state = None
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for signals, labels in loaders["train"]:
            signals = signals.to(device)
            optimizer.zero_grad()
            output = model(signals)
            loss = model.compute_loss(signals, output)[0]
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step(np.mean(epoch_losses))

        val_scores, val_labels = compute_anomaly_scores(model, loaders["val"], device)
        val_threshold = find_threshold(model, loaders["val_normal"], device)
        val_result = compute_metrics(val_labels, val_scores, val_threshold)

        if val_result.auroc > best_val_auroc:
            best_val_auroc = val_result.auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.6f} | "
                        f"Val AUROC: {val_result.auroc:.4f}")

    train_time = time.time() - train_start
    model.load_state_dict(best_state)

    threshold = find_threshold(model, loaders["val_normal"], device)

    all_test_scores = compute_anomaly_scores_raw(model, test_signals, device)

    normal_scores = all_test_scores[normal_mask]

    overall_result = compute_metrics(test_labels, all_test_scores, threshold)
    logger.info(f"    Overall test: {overall_result}")

    perclass_results = {"overall": overall_result}

    for cls, cls_mask in class_masks.items():
        if cls_mask.sum() == 0:
            logger.warning(f"    {cls}: no samples found, skipping")
            continue

        cls_scores = np.concatenate([normal_scores, all_test_scores[cls_mask]])
        cls_labels = np.concatenate([
            np.zeros(normal_mask.sum(), dtype=int),
            np.ones(cls_mask.sum(), dtype=int),
        ])

        cls_result = compute_metrics(cls_labels, cls_scores, threshold)
        perclass_results[cls] = cls_result
        logger.info(f"    {cls} (n={cls_mask.sum()}): {cls_result}")

    return perclass_results, model.model_size_mb(), train_time



def main():
    parser = argparse.ArgumentParser(description="Per-Class Breakdown")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl")
    parser.add_argument("--raw_dir", type=str, default="data/ptb-xl-raw",
                        help="Path to original PTB-XL with ptbxl_database.csv")
    parser.add_argument("--model", type=str, default=None,
                        choices=["vanilla_ae", "conv_ae"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    setup_logging()
    device = get_device()
    logger.info(f"Device: {device}")

    splits = load_splits(args.data_dir)
    loaders = create_dataloaders(splits, batch_size=args.batch_size)

    logger.info("Building per-class masks from PTB-XL metadata...")
    class_masks, normal_mask = build_perclass_masks(args.raw_dir, args.data_dir)

    test_signals = splits["test"].signals  # already a tensor
    test_labels_np = splits["test"].labels.numpy()

    models_to_run = [args.model] if args.model else ["vanilla_ae", "conv_ae"]
    seeds = args.seeds or SEEDS

    os.makedirs("outputs", exist_ok=True)
    csv_logger = ResultLogger("outputs/ablation_perClass.csv",
                              extra_columns=["bottleneck", "condition"])

    for model_name in models_to_run:
        for seed in seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"{model_name} | seed={seed} | Per-class breakdown")
            logger.info(f"{'='*60}")

            perclass_results, size_mb, train_time = train_and_evaluate_perclass(
                model_name, loaders, seed, device,
                test_signals, test_labels_np, class_masks, normal_mask,
                threshold_from_val=True,
                epochs=args.epochs, lr=args.lr, bottleneck=args.bottleneck,
            )

            for condition, result in perclass_results.items():
                csv_logger.log(
                    model=model_name,
                    setting="centralised",
                    bottleneck=args.bottleneck,
                    condition=condition,
                    beta="",
                    epsilon="",
                    precision_type="fp32",
                    seed=seed,
                    auroc=result.auroc,
                    auprc=result.auprc,
                    sensitivity=result.sensitivity,
                    specificity=result.specificity,
                    precision_score=result.precision,
                    f1=result.f1,
                    model_size_mb=size_mb,
                    flops_m="",
                    inference_latency_ms="",
                    peak_memory_mb="",
                    training_time_s=train_time,
                )

    logger.info(f"\n{'='*60}")
    logger.info("PER-CLASS SUMMARY")
    logger.info(f"{'='*60}")

    import pandas as pd
    df = pd.read_csv("outputs/ablation_perClass.csv")
    perclass_df = df[df["condition"].notna() & (df["condition"] != "")]
    if len(perclass_df) > 0:
        summary = perclass_df.groupby(["model", "condition"]).agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        ).round(4)
        logger.info(f"\n{summary}")

    logger.info(f"\nResults saved to outputs/ablation_perClass.csv")


if __name__ == "__main__":
    main()