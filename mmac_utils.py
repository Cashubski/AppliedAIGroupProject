"""
shared utilities for the MTL and uncertainty notebooks

As per Category 2, AI was used to proofread code
"""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# paths + constants
PROJECT_ROOT: Path = Path.cwd()
DATA_ROOT: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_ROOT / "Training" / "Training_Images"
TRAIN_CSV: Path = DATA_ROOT / "Training" / "Training_LabelsDemographic.csv"
TEST_DIR: Path = DATA_ROOT / "Testing" / "Testing_Images"
TEST_CSV: Path = DATA_ROOT / "Testing" / "Testing_LabelDemographic.csv"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" 
ARTEFACT_DIR: Path = PROJECT_ROOT / "artefacts"
BASELINE_OUTPUT_DIR: Path = OUTPUT_DIR / "baseline"
BASELINE_CKPT: Path = CHECKPOINT_DIR / "baseline_resnet50_best.pt"

NUM_CLASSES: int = 5
CLASS_NAMES: Tuple[str, ...] = (
    "0: No macular lesions", "1: Tessellated fundus", "2: Diffuse atrophy",
    "3: Patchy atrophy", "4: Macular atrophy",
)
LABEL_COL: str = "myopic_maculopathy_grade"
IMAGE_COL: str = "image"
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

_SCALAR_METRICS: Tuple[str, ...] = (
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "quadratic_kappa", "macro_auroc",
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


# helpers 
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_cuda_optimizations() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# transforms + data
def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_label_frame(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    # drop rows whose image file is missing on disk
    exists = frame[IMAGE_COL].apply(lambda n: (Path(image_dir) / n).is_file())
    if (~exists).any():
        print(f"[data] dropping {(~exists).sum()} missing rows in {image_dir}")
        frame = frame[exists].reset_index(drop=True)
    return frame


def stratified_split(frame: pd.DataFrame, val_fraction: float, seed: int):
    tr, va = train_test_split(
        np.arange(len(frame)), test_size=val_fraction,
        random_state=seed, shuffle=True, stratify=frame[LABEL_COL].values,
    )
    return frame.iloc[tr].reset_index(drop=True), frame.iloc[va].reset_index(drop=True)


@dataclass
class MMACClassificationDataset(Dataset):
    """simple (image, grade) dataset used by baseline + uncertainty notebooks."""
    frame: pd.DataFrame
    image_dir: Path
    transform: Optional[transforms.Compose] = None

    def __post_init__(self) -> None:
        self.frame = self.frame.reset_index(drop=True)
        self.image_dir = Path(self.image_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(self.image_dir / row[IMAGE_COL]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row[LABEL_COL])


def build_dataloaders(image_size: int, batch_size: int, num_workers: int,
                      val_split: float, seed: int) -> Dict[str, Any]:
    tr_frame = load_label_frame(TRAIN_CSV, TRAIN_DIR)
    te_frame = load_label_frame(TEST_CSV, TEST_DIR)
    tr_frame, va_frame = stratified_split(tr_frame, val_split, seed)
    tr_tfm, ev_tfm = build_train_transform(image_size), build_eval_transform(image_size)
    ds = {
        "train": MMACClassificationDataset(tr_frame, TRAIN_DIR, tr_tfm),
        "val":   MMACClassificationDataset(va_frame, TRAIN_DIR, ev_tfm),
        "test":  MMACClassificationDataset(te_frame, TEST_DIR,  ev_tfm),
    }
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(ds["train"], shuffle=True,  drop_last=True,  **common),
        "val":   DataLoader(ds["val"],   shuffle=False, drop_last=False, **common),
        "test":  DataLoader(ds["test"],  shuffle=False, drop_last=False, **common),
        "train_ds": ds["train"], "val_ds": ds["val"], "test_ds": ds["test"],
    }


# metrics + bootstrap
@dataclass
class MetricBundle:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    quadratic_kappa: float
    macro_auroc: Optional[float]
    per_class_precision: List[float] = field(default_factory=list)
    per_class_recall: List[float] = field(default_factory=list)
    per_class_f1: List[float] = field(default_factory=list)
    per_class_support: List[int] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def pretty(self) -> str:
        a = f"{self.macro_auroc:.4f}" if (self.macro_auroc is not None
                                          and np.isfinite(self.macro_auroc)) else "  n/a"
        return (f"acc={self.accuracy:.4f}  bal_acc={self.balanced_accuracy:.4f}  "
                f"macroF1={self.macro_f1:.4f}  kappa={self.quadratic_kappa:.4f}  AUROC={a}")


def compute_metrics(y_true, y_pred, y_prob=None,
                    num_classes: int = NUM_CLASSES) -> MetricBundle:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))
    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    auroc = None
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        try:
            auroc = roc_auc_score(y_true, np.asarray(y_prob), multi_class="ovr",
                                  average="macro", labels=labels)
            if not np.isfinite(auroc): auroc = None
        except ValueError:
            auroc = None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        quadratic_kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        macro_auroc=float(auroc) if auroc is not None else None,
        per_class_precision=prec.astype(float).tolist(),
        per_class_recall=rec.astype(float).tolist(),
        per_class_f1=f1c.astype(float).tolist(),
        per_class_support=sup.astype(int).tolist(),
        confusion_matrix=cm.astype(int).tolist(),
    )


def _stratified_bootstrap_index(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # sample with replacement within each class so rare grades stay represented
    out = np.empty_like(y_true, dtype=np.int64)
    pos = 0
    for c in np.unique(y_true):
        cls = np.flatnonzero(y_true == c)
        out[pos: pos + len(cls)] = cls[rng.integers(0, len(cls), size=len(cls))]
        pos += len(cls)
    return out


def bootstrap_metrics(y_true, y_pred, y_prob=None, *, num_classes=NUM_CLASSES,
                      n_resamples=1000, ci_level=0.95, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_prob is not None: y_prob = np.asarray(y_prob)
    point = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes).to_dict()
    samples: Dict[str, List[float]] = {m: [] for m in _SCALAR_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        idx = _stratified_bootstrap_index(y_true, rng)
        b = compute_metrics(y_true[idx], y_pred[idx],
                            y_prob[idx] if y_prob is not None else None,
                            num_classes=num_classes)
        for m in _SCALAR_METRICS:
            v = getattr(b, m)
            if v is not None and np.isfinite(v):
                samples[m].append(float(v))
    a = (1.0 - ci_level) / 2.0
    results: Dict[str, Dict[str, float]] = {}
    for m in _SCALAR_METRICS:
        arr = np.asarray(samples[m], dtype=float)
        pv = point.get(m); pv = float(pv) if pv is not None else float("nan")
        if arr.size == 0:
            results[m] = {"point": pv, "ci_low": float("nan"),
                          "ci_high": float("nan"), "std": float("nan")}
        else:
            lo, hi = np.quantile(arr, [a, 1.0 - a])
            results[m] = {"point": pv, "ci_low": float(lo), "ci_high": float(hi),
                          "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}
    return results


def paired_bootstrap_compare(y_true, y_pred_a, y_pred_b, y_prob_a=None, y_prob_b=None, *,
                             num_classes=NUM_CLASSES, metric="quadratic_kappa",
                             n_resamples=2000, ci_level=0.95, seed=42):
    """paired bootstrap delta on the SAME resamples so the comparison is honest."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int); y_pred_b = np.asarray(y_pred_b).astype(int)
    if y_prob_a is not None: y_prob_a = np.asarray(y_prob_a)
    if y_prob_b is not None: y_prob_b = np.asarray(y_prob_b)

    def _scalar(y, p, q):
        v = getattr(compute_metrics(y, p, q, num_classes=num_classes), metric)
        return float(v) if v is not None and np.isfinite(v) else np.nan

    ma = _scalar(y_true, y_pred_a, y_prob_a)
    mb = _scalar(y_true, y_pred_b, y_prob_b)
    delta = ma - mb
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    for _ in range(n_resamples):
        idx = _stratified_bootstrap_index(y_true, rng)
        va = _scalar(y_true[idx], y_pred_a[idx], y_prob_a[idx] if y_prob_a is not None else None)
        vb = _scalar(y_true[idx], y_pred_b[idx], y_prob_b[idx] if y_prob_b is not None else None)
        if np.isfinite(va) and np.isfinite(vb):
            deltas.append(va - vb)
    arr = np.asarray(deltas, dtype=float)
    a = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(arr, [a, 1.0 - a]) if arr.size else (np.nan, np.nan)
    # two-sided p-value via proportion of resamples on the wrong side of zero
    if arr.size == 0 or not np.isfinite(delta):
        p = float("nan")
    else:
        tail = (arr <= 0).mean() if delta >= 0 else (arr >= 0).mean()
        p = min(1.0, 2.0 * float(tail))
    return {"metric_a": ma, "metric_b": mb, "delta": delta,
            "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "n_resamples": int(arr.size)}


def format_ci(ci: Dict[str, float], digits: int = 4) -> str:
    p, lo, hi = ci.get("point", float("nan")), ci.get("ci_low", float("nan")), ci.get("ci_high", float("nan"))
    if not (np.isfinite(p) and np.isfinite(lo) and np.isfinite(hi)):
        return f"{p:.{digits}f}  [n/a]"
    return f"{p:.{digits}f}  [{lo:.{digits}f}, {hi:.{digits}f}]"


# checkpoints + history 
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def save_checkpoint(path: Path, *, model: nn.Module, optimizer=None, scheduler=None,
                    epoch: int, metrics=None, extra=None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"epoch": int(epoch),
                               "model_state": unwrap_model(model).state_dict(),
                               "metrics": metrics or {}, "extra": extra or {}}
    if optimizer is not None: payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, *, model=None, optimizer=None, scheduler=None,
                    map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if model is not None and "model_state" in payload:
        unwrap_model(model).load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metrics: List[Dict[str, Any]] = field(default_factory=list)
    val_metrics: List[Dict[str, Any]] = field(default_factory=list)
    learning_rates: List[List[float]] = field(default_factory=list)
    # per-epoch recording (aux losses, kl weight, log_vars...)
    extra: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, **kw) -> None:
        for k, v in kw.items():
            getattr(self, k).append(v)

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.__dict__, f, indent=2, default=_json_default)


# model + warm start
class MMACResNet50(nn.Module):
    """ImageNet pretrained ResNet-50 + Linear->BN->ReLU->Dropout->Linear head."""

    def __init__(self, num_classes: int = NUM_CLASSES,
                 pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()   # drop ImageNet head
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        # kaiming init for the fresh head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))

    def parameter_groups(self, backbone_lr, head_lr, weight_decay):
        return [
            {"params": list(self.backbone.parameters()), "lr": backbone_lr,
             "weight_decay": weight_decay, "name": "backbone"},
            {"params": list(self.head.parameters()), "lr": head_lr,
             "weight_decay": weight_decay, "name": "head"},
        ]


def warm_start_from_ckpt(model: nn.Module, ckpt_path: Path, *,
                         backbone_only: bool = False) -> None:
    """load a baseline checkpoint into `model`. `backbone_only` skips head keys."""
    p = Path(ckpt_path)
    if not p.is_file():
        print(f"[warm] no checkpoint at {p} - skipping")
        return
    state = torch.load(p, map_location="cpu")["model_state"]
    if backbone_only:
        state = {k: v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    tag = "backbone" if backbone_only else "full"
    print(f"[warm] {tag} load from {p.name} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


# amp + scheduler helpers
def amp_scaler(use_amp: bool, device: torch.device):
    return torch.amp.GradScaler(device=device.type) if (use_amp and device.type == "cuda") else None

def amp_ctx(use_amp: bool, device: torch.device):
    return torch.amp.autocast(device_type=device.type) if (use_amp and device.type == "cuda") else nullcontext()

def build_cosine_with_warmup(optimizer: Optimizer, total_epochs: int, warmup_epochs: int):
    base = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=0.0)
    if warmup_epochs <= 0:
        return base
    warm = LambdaLR(optimizer, lr_lambda=lambda e: float(e + 1) / max(1, warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warm, base], milestones=[warmup_epochs])


# plotting 
def plot_confusion_matrix(cm, class_names, normalize=True, save_path=None,
                          title="Confusion matrix"):
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt = ".2f"
    else:
        disp, fmt = cm, ".0f"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
