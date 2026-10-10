# -*- coding: utf-8 -*-
"""
Multi-modal + Multi-task ResNet1D for signal quality assessment.

Goal:
    Input  : ECG + PPG + PCG raw waveforms, stacked as 3 channels: X [N, 3, 5000]
    Output : three classification heads:
             1) ECG quality: 0/1/2
             2) PPG quality: 0/1/2
             3) PCG quality: 0/1/2

Recommended data preparation:
    Run your Data_save script three times, once for each modality label set.
    The script must save Basekey_for{TYPE}.npy and Segment_for{TYPE}.npy.

Default expected files in data_dir:
    ECG task:
        ECG_Data_forECG_251124.npy
        ECG_251124_Label.npy
        Basekey_forECG_251124.npy
        Segment_forECG_251124.npy

    PPG task:
        PPG_Data_forPPG_251124.npy
        PPG_251124_Label.npy
        Basekey_forPPG_251124.npy
        Segment_forPPG_251124.npy

    PCG task:
        PCG_Data_forPCG_251124.npy
        PCG_251124_Label.npy
        Basekey_forPCG_251124.npy
        Segment_forPCG_251124.npy

The loader aligns samples by (basekey, segment), then uses only the intersection
that has ECG label, PPG label and PCG label simultaneously.

Run directly:
    python train_raw_3signal_resnet1d_mtl.py

If your TYPE names are different, edit DEFAULT_CONFIG below or override via CLI:
    python train_raw_3signal_resnet1d_mtl.py --ecg_type ECG_xxx --ppg_type PPG_xxx --pcg_type PCG_xxx
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 0. Default configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data directory and task type names.
    "data_dir": "./",
    "ecg_type": "ECG_251124",
    "ppg_type": "PPG_251124",
    "pcg_type": "PCG_251124",
    "out_dir": "./raw_3signal_resnet1d_mtl_results",

    # Split by basekey to avoid leakage.
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42,

    # Model.
    "num_classes": 3,
    "base_channels": 64,
    "dropout": 0.2,

    # Task loss weights.
    # Usually keep all 1.0 for a clean baseline.
    "ecg_loss_weight": 1.0,
    "ppg_loss_weight": 1.0,
    "pcg_loss_weight": 1.0,

    # Training.
    "epochs": 100,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,
    "num_workers": 4,
    "device": "cuda:1",
    "amp": True,
    "sample_zscore": False,

    # Feature output.
    "save_features": True,
}

TASKS = ["ecg", "ppg", "pcg"]
TASK_TO_SIGNAL_FILE_PREFIX = {
    "ecg": "ECG_Data_for",
    "ppg": "PPG_Data_for",
    "pcg": "PCG_Data_for",
}


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. Data loading and alignment
# ============================================================
def _make_key(basekey: str, segment: int) -> str:
    return f"{basekey}__SEG{int(segment)}"


def load_one_task_records(data_dir: Path, task: str, type_name: str) -> Dict[str, dict]:
    """
    Load one modality-task records.

    For the ECG task, this function loads ECG waveform + ECG label.
    For the PPG task, it loads PPG waveform + PPG label.
    For the PCG task, it loads PCG waveform + PCG label.

    Returns:
        records[key] = {
            "signal": np.ndarray [L],
            "label": int,
            "basekey": str,
            "segment": int,
        }
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")

    signal_path = data_dir / f"{TASK_TO_SIGNAL_FILE_PREFIX[task]}{type_name}.npy"
    label_path = data_dir / f"{type_name}_Label.npy"
    basekey_path = data_dir / f"Basekey_for{type_name}.npy"
    segment_path = data_dir / f"Segment_for{type_name}.npy"

    required = [signal_path, label_path, basekey_path, segment_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files for {task.upper()} task, type_name={type_name}:\n" + "\n".join(missing)
        )

    signal = np.load(signal_path).astype(np.float32)
    label = np.load(label_path).squeeze().astype(np.int64)
    basekeys = np.load(basekey_path).astype(str)
    segments = np.load(segment_path).astype(np.int64)

    if signal.ndim != 2:
        raise ValueError(f"Expected {signal_path.name} to be [N, L], got {signal.shape}")

    n = len(signal)
    if not (len(label) == len(basekeys) == len(segments) == n):
        raise ValueError(
            f"Length mismatch for {task.upper()} task: signal={n}, label={len(label)}, "
            f"basekeys={len(basekeys)}, segments={len(segments)}"
        )

    records = {}
    duplicate_count = 0
    for i in range(n):
        k = _make_key(basekeys[i], segments[i])
        if k in records:
            duplicate_count += 1
            # Keep the first one. Duplicates usually indicate repeated tif names or duplicate segments.
            continue
        records[k] = {
            "signal": signal[i],
            "label": int(label[i]),
            "basekey": basekeys[i],
            "segment": int(segments[i]),
        }

    print(
        f"Loaded {task.upper()} task | type={type_name} | records={len(records)} | "
        f"duplicates_skipped={duplicate_count} | signal_len={signal.shape[1]} | "
        f"label_dist={dict(zip(*np.unique(label, return_counts=True)))}"
    )
    return records


def load_aligned_multitask_data(data_dir: str, ecg_type: str, ppg_type: str, pcg_type: str):
    """
    Align ECG/PPG/PCG task datasets by (basekey, segment).

    Returns:
        X       : [N, 3, L]
        y       : [N, 3], columns = [ecg_label, ppg_label, pcg_label]
        basekeys: [N]
        segments: [N]
    """
    data_dir = Path(data_dir)

    ecg_records = load_one_task_records(data_dir, "ecg", ecg_type)
    ppg_records = load_one_task_records(data_dir, "ppg", ppg_type)
    pcg_records = load_one_task_records(data_dir, "pcg", pcg_type)

    ecg_keys = set(ecg_records.keys())
    ppg_keys = set(ppg_records.keys())
    pcg_keys = set(pcg_records.keys())
    common_keys = sorted(ecg_keys & ppg_keys & pcg_keys)

    if len(common_keys) == 0:
        raise ValueError(
            "No common samples found across ECG/PPG/PCG labels.\n"
            "Please check whether Basekey_for*.npy and Segment_for*.npy refer to the same recordings and segments."
        )

    n_ecg_only = len(ecg_keys - (ppg_keys & pcg_keys))
    n_ppg_only = len(ppg_keys - (ecg_keys & pcg_keys))
    n_pcg_only = len(pcg_keys - (ecg_keys & ppg_keys))
    print("=" * 90)
    print("Alignment summary by (basekey, segment)")
    print(f"ECG records: {len(ecg_keys)}")
    print(f"PPG records: {len(ppg_keys)}")
    print(f"PCG records: {len(pcg_keys)}")
    print(f"Common records used for multitask learning: {len(common_keys)}")
    print(f"Excluded ECG-side non-common records: {n_ecg_only}")
    print(f"Excluded PPG-side non-common records: {n_ppg_only}")
    print(f"Excluded PCG-side non-common records: {n_pcg_only}")
    print("=" * 90)

    X_list, y_list, basekey_list, segment_list = [], [], [], []
    for k in common_keys:
        ecg_sig = ecg_records[k]["signal"]
        ppg_sig = ppg_records[k]["signal"]
        pcg_sig = pcg_records[k]["signal"]

        if not (len(ecg_sig) == len(ppg_sig) == len(pcg_sig)):
            raise ValueError(
                f"Signal length mismatch at key={k}: ECG={len(ecg_sig)}, PPG={len(ppg_sig)}, PCG={len(pcg_sig)}"
            )

        x = np.stack([ecg_sig, ppg_sig, pcg_sig], axis=0).astype(np.float32)  # [3, L]
        y = np.array([
            ecg_records[k]["label"],
            ppg_records[k]["label"],
            pcg_records[k]["label"],
        ], dtype=np.int64)

        if not np.isfinite(x).all():
            continue

        X_list.append(x)
        y_list.append(y)
        basekey_list.append(ecg_records[k]["basekey"])
        segment_list.append(ecg_records[k]["segment"])

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, 3, L]
    y = np.stack(y_list, axis=0).astype(np.int64)    # [N, 3]
    basekeys = np.array(basekey_list).astype(str)
    segments = np.array(segment_list, dtype=np.int64)

    print("=" * 90)
    print("Loaded aligned multimodal multitask dataset")
    print(f"X shape: {X.shape}  # [N, 3, L]")
    print(f"y shape: {y.shape}  # [N, 3] = [ECG_label, PPG_label, PCG_label]")
    print(f"unique basekeys: {len(np.unique(basekeys))}")
    for j, task in enumerate(TASKS):
        print(f"{task.upper()} label distribution:", dict(zip(*np.unique(y[:, j], return_counts=True))))
    print("=" * 90)

    return X, y, basekeys, segments


# ============================================================
# 3. Group split
# ============================================================
def group_train_val_test_split(
    y: np.ndarray,
    groups: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    indices = np.arange(len(groups))

    # Use ECG label for stratification-like input to GroupShuffleSplit only.
    # GroupShuffleSplit itself does not enforce exact class stratification.
    y_for_split = y[:, 0]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(indices, y_for_split, groups))

    val_size_within_trainval = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_trainval, random_state=seed + 1)
    tr_rel_idx, val_rel_idx = next(
        gss_val.split(trainval_idx, y_for_split[trainval_idx], groups[trainval_idx])
    )

    train_idx = trainval_idx[tr_rel_idx]
    val_idx = trainval_idx[val_rel_idx]
    return train_idx, val_idx, test_idx


def save_split_info(out_dir, y, groups, train_idx, val_idx, test_idx):
    out_dir = Path(out_dir)

    def summarize(name, idx):
        row = {
            "split": name,
            "n_samples": int(len(idx)),
            "n_groups": int(len(np.unique(groups[idx]))),
        }
        for task_i, task in enumerate(TASKS):
            values, counts = np.unique(y[idx, task_i], return_counts=True)
            for v, c in zip(values, counts):
                row[f"{task}_class_{int(v)}"] = int(c)
        return row

    split_df = pd.DataFrame([
        summarize("train", train_idx),
        summarize("val", val_idx),
        summarize("test", test_idx),
    ])
    split_df.to_csv(out_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    overlap = {
        "train_val_overlap": len(train_groups & val_groups),
        "train_test_overlap": len(train_groups & test_groups),
        "val_test_overlap": len(val_groups & test_groups),
    }
    with open(out_dir / "group_overlap_check.json", "w", encoding="utf-8") as f:
        json.dump(overlap, f, indent=2, ensure_ascii=False)

    print(split_df)
    print("Group overlap check:", overlap)


# ============================================================
# 4. Dataset
# ============================================================
class MultiTaskRawSignalDataset(Dataset):
    def __init__(self, X, y, sample_zscore: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.sample_zscore = sample_zscore

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.sample_zscore:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        return torch.from_numpy(x), torch.from_numpy(self.y[idx]).long()


# ============================================================
# 5. ResNet1D backbone + three task heads
# ============================================================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class MultiTaskResNet1D(nn.Module):
    """
    Input:
        x: [B, 3, 5000]

    Outputs:
        logits_dict = {
            "ecg": [B, 3],
            "ppg": [B, 3],
            "pcg": [B, 3],
        }

    The shared GAP vector is the multimodal deep feature.
    """
    def __init__(self, in_channels=3, num_classes=3, base_channels=64, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks=2, stride=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)

        feature_dim = base_channels * 8
        self.dropout = nn.Dropout(dropout)
        self.ecg_head = nn.Linear(feature_dim, num_classes)
        self.ppg_head = nn.Linear(feature_dim, num_classes)
        self.pcg_head = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride, dropout):
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.gap(x).squeeze(-1)
        return features

    def forward(self, x, return_features=False):
        features = self.forward_features(x)
        h = self.dropout(features)
        logits = {
            "ecg": self.ecg_head(h),
            "ppg": self.ppg_head(h),
            "pcg": self.pcg_head(h),
        }
        if return_features:
            return logits, features
        return logits


# ============================================================
# 6. Loss, metrics, train/evaluate
# ============================================================
def make_class_weights(y_train, task_index: int, num_classes=3):
    counts = np.bincount(y_train[:, task_index], minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_criterions(y_train, num_classes, device):
    criterions = {}
    for idx, task in enumerate(TASKS):
        weights = make_class_weights(y_train, idx, num_classes=num_classes).to(device)
        print(f"{task.upper()} class weights:", weights.detach().cpu().numpy())
        criterions[task] = nn.CrossEntropyLoss(weight=weights)
    return criterions


def compute_multitask_loss(logits_dict, y, criterions, task_loss_weights):
    losses = {}
    total_loss = 0.0
    for i, task in enumerate(TASKS):
        loss_i = criterions[task](logits_dict[task], y[:, i])
        losses[task] = loss_i
        total_loss = total_loss + task_loss_weights[task] * loss_i
    return total_loss, losses


def task_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def aggregate_metrics(y_true_matrix, y_pred_matrix, avg_loss):
    metrics = {"loss": avg_loss}
    macro_f1s = []
    accuracies = []
    for i, task in enumerate(TASKS):
        m = task_metrics(y_true_matrix[:, i], y_pred_matrix[:, i], task)
        metrics.update(m)
        macro_f1s.append(m[f"{task}_macro_f1"])
        accuracies.append(m[f"{task}_accuracy"])
    metrics["mean_macro_f1"] = float(np.mean(macro_f1s))
    metrics["mean_accuracy"] = float(np.mean(accuracies))
    return metrics


def run_one_epoch(model, loader, criterions, optimizer, device, task_loss_weights, scaler=None):
    model.train()
    total_loss_sum = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss, _ = compute_multitask_loss(logits, y, criterions, task_loss_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss, _ = compute_multitask_loss(logits, y, criterions, task_loss_weights)
            loss.backward()
            optimizer.step()

        preds = torch.stack([logits[task].detach().argmax(dim=1) for task in TASKS], dim=1)
        total_loss_sum += loss.item() * x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss_sum / len(loader.dataset)
    return aggregate_metrics(y_true, y_pred, avg_loss)


@torch.no_grad()
def evaluate(model, loader, criterions, device, task_loss_weights, save_features=False):
    model.eval()
    total_loss_sum = 0.0
    all_preds = []
    all_targets = []
    all_probs = {task: [] for task in TASKS}
    all_features = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if save_features:
            logits, features = model(x, return_features=True)
            all_features.append(features.cpu().numpy())
        else:
            logits = model(x)

        loss, _ = compute_multitask_loss(logits, y, criterions, task_loss_weights)
        preds = torch.stack([logits[task].argmax(dim=1) for task in TASKS], dim=1)

        total_loss_sum += loss.item() * x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        for task in TASKS:
            all_probs[task].append(torch.softmax(logits[task], dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    probs = {task: np.concatenate(all_probs[task], axis=0) for task in TASKS}
    features = np.concatenate(all_features, axis=0) if save_features and all_features else None
    avg_loss = total_loss_sum / len(loader.dataset)
    metrics = aggregate_metrics(y_true, y_pred, avg_loss)
    return metrics, y_true, y_pred, probs, features


# ============================================================
# 7. Save outputs
# ============================================================
def save_eval_outputs(out_dir, split_name, y_true, y_pred, probs, metrics, features=None):
    out_dir = Path(out_dir)

    pd.DataFrame([{"split": split_name, **metrics}]).to_csv(
        out_dir / f"{split_name}_metrics.csv", index=False, encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame()
    for i, task in enumerate(TASKS):
        pred_df[f"{task}_true"] = y_true[:, i]
        pred_df[f"{task}_pred"] = y_pred[:, i]
        pred_df[f"{task}_prob_class0"] = probs[task][:, 0]
        pred_df[f"{task}_prob_class1"] = probs[task][:, 1]
        pred_df[f"{task}_prob_class2"] = probs[task][:, 2]
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    for i, task in enumerate(TASKS):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"])
        cm_df.to_csv(out_dir / f"{split_name}_{task}_confusion_matrix.csv", encoding="utf-8-sig")

        report = classification_report(
            y_true[:, i], y_pred[:, i], labels=[0, 1, 2], output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(
            out_dir / f"{split_name}_{task}_classification_report.csv", encoding="utf-8-sig"
        )

    if features is not None:
        np.save(out_dir / f"{split_name}_shared_deep_features.npy", features)


def save_alignment_table(out_dir, basekeys, segments, y):
    df = pd.DataFrame({
        "basekey": basekeys,
        "segment": segments,
        "ecg_label": y[:, 0],
        "ppg_label": y[:, 1],
        "pcg_label": y[:, 2],
    })
    df.to_csv(Path(out_dir) / "aligned_samples.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 8. Main
# ============================================================
def main(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    X, y, basekeys, segments = load_aligned_multitask_data(
        data_dir=args.data_dir,
        ecg_type=args.ecg_type,
        ppg_type=args.ppg_type,
        pcg_type=args.pcg_type,
    )
    save_alignment_table(out_dir, basekeys, segments, y)

    train_idx, val_idx, test_idx = group_train_val_test_split(
        y=y,
        groups=basekeys,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_split_info(out_dir, y, basekeys, train_idx, val_idx, test_idx)

    train_ds = MultiTaskRawSignalDataset(X[train_idx], y[train_idx], sample_zscore=args.sample_zscore)
    val_ds = MultiTaskRawSignalDataset(X[val_idx], y[val_idx], sample_zscore=args.sample_zscore)
    test_ds = MultiTaskRawSignalDataset(X[test_idx], y[test_idx], sample_zscore=args.sample_zscore)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"Using device: {device}")

    model = MultiTaskResNet1D(
        in_channels=3,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    criterions = build_criterions(y[train_idx], args.num_classes, device)
    task_loss_weights = {
        "ecg": args.ecg_loss_weight,
        "ppg": args.ppg_loss_weight,
        "pcg": args.pcg_loss_weight,
    }
    print("Task loss weights:", task_loss_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val_mean_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model, train_loader, criterions, optimizer, device, task_loss_weights, scaler=scaler
        )
        val_metrics, _, _, _, _ = evaluate(
            model, val_loader, criterions, device, task_loss_weights, save_features=False
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False, encoding="utf-8-sig")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_meanF1={train_metrics['mean_macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_meanF1={val_metrics['mean_macro_f1']:.4f} | "
            f"val_ecgF1={val_metrics['ecg_macro_f1']:.4f}, "
            f"val_ppgF1={val_metrics['ppg_macro_f1']:.4f}, "
            f"val_pcgF1={val_metrics['pcg_macro_f1']:.4f}"
        )

        if val_metrics["mean_macro_f1"] > best_val_mean_macro_f1:
            best_val_mean_macro_f1 = val_metrics["mean_macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mean_macro_f1": best_val_mean_macro_f1,
                    "args": vars(args),
                },
                out_dir / "best_multitask_resnet1d.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val mean macro-F1 = {best_val_mean_macro_f1:.4f}"
            )
            break

    checkpoint = torch.load(out_dir / "best_multitask_resnet1d.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, y_true_val, y_pred_val, probs_val, val_features = evaluate(
        model, val_loader, criterions, device, task_loss_weights, save_features=args.save_features
    )
    test_metrics, y_true_test, y_pred_test, probs_test, test_features = evaluate(
        model, test_loader, criterions, device, task_loss_weights, save_features=args.save_features
    )

    save_eval_outputs(out_dir, "val", y_true_val, y_pred_val, probs_val, val_metrics, val_features)
    save_eval_outputs(out_dir, "test", y_true_test, y_pred_test, probs_test, test_metrics, test_features)

    summary = pd.DataFrame([
        {"split": "val", **val_metrics, "best_epoch": best_epoch},
        {"split": "test", **test_metrics, "best_epoch": best_epoch},
    ])
    summary.to_csv(out_dir / "final_metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("Multi-task training finished.")
    print(f"Best epoch: {best_epoch}")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print(f"All outputs saved to: {out_dir.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--ecg_type", type=str, default=DEFAULT_CONFIG["ecg_type"])
    parser.add_argument("--ppg_type", type=str, default=DEFAULT_CONFIG["ppg_type"])
    parser.add_argument("--pcg_type", type=str, default=DEFAULT_CONFIG["pcg_type"])
    parser.add_argument("--out_dir", type=str, default=DEFAULT_CONFIG["out_dir"])

    parser.add_argument("--train_ratio", type=float, default=DEFAULT_CONFIG["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_CONFIG["val_ratio"])
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_CONFIG["test_ratio"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    parser.add_argument("--num_classes", type=int, default=DEFAULT_CONFIG["num_classes"])
    parser.add_argument("--base_channels", type=int, default=DEFAULT_CONFIG["base_channels"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])

    parser.add_argument("--ecg_loss_weight", type=float, default=DEFAULT_CONFIG["ecg_loss_weight"])
    parser.add_argument("--ppg_loss_weight", type=float, default=DEFAULT_CONFIG["ppg_loss_weight"])
    parser.add_argument("--pcg_loss_weight", type=float, default=DEFAULT_CONFIG["pcg_loss_weight"])

    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])

    parser.add_argument("--amp", dest="amp", action="store_true", default=DEFAULT_CONFIG["amp"])
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--sample_zscore", action="store_true", default=DEFAULT_CONFIG["sample_zscore"])
    parser.add_argument("--save_features", dest="save_features", action="store_true", default=DEFAULT_CONFIG["save_features"])
    parser.add_argument("--no_save_features", dest="save_features", action="store_false")

    args = parser.parse_args()

    print("=" * 90)
    print("Using default/in-code configuration. Final args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 90)

    main(args)


    '''
    Multi-task training finished.
Best epoch: 23
Validation metrics: {'loss': 0.8903775457787303, 'ecg_accuracy': 0.9048672566371682, 'ecg_macro_f1': 0.9008338744543446, 'ecg_weighted_f1': 0.9060124758077661, 'ecg_macro_precision': 0.8946908862189648, 'ecg_macro_recall': 0.9092788443529467, 'ppg_accuracy': 0.9192477876106194, 'ppg_macro_f1': 0.846883560510023, 'ppg_weighted_f1': 0.9254029874861983, 'ppg_macro_precision': 0.8300954662566444, 'ppg_macro_recall': 0.8816478288795517, 'pcg_accuracy': 0.9021017699115044, 'pcg_macro_f1': 0.8089542032929828, 'pcg_weighted_f1': 0.9034047708947033, 'pcg_macro_precision': 0.7876746833863623, 'pcg_macro_recall': 0.8347096965069573, 'mean_macro_f1': 0.8522238794191169, 'mean_accuracy': 0.9087389380530974}
Test metrics: {'loss': 1.2371674907886414, 'ecg_accuracy': 0.8842996408414572, 'ecg_macro_f1': 0.8748054824522692, 'ecg_weighted_f1': 0.8844265901358233, 'ecg_macro_precision': 0.8777433882713184, 'ecg_macro_recall': 0.8721334379905809, 'ppg_accuracy': 0.9143150333504361, 'ppg_macro_f1': 0.8361697541731412, 'ppg_weighted_f1': 0.9233261522022591, 'ppg_macro_precision': 0.8145294511999399, 'ppg_macro_recall': 0.8852512468680193, 'pcg_accuracy': 0.913801949717804, 'pcg_macro_f1': 0.8095946641855747, 'pcg_weighted_f1': 0.9144385768468828, 'pcg_macro_precision': 0.7739236614185905, 'pcg_macro_recall': 0.8714662545444208, 'mean_macro_f1': 0.840189966936995, 'mean_accuracy': 0.9041388746365658}
    '''
