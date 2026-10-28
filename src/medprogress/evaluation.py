"""Leakage-aware model training, evaluation and missing-modality stress tests."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    CLINICAL_COLS,
    FEATURE_COLS,
    IMAGE_COLS,
    MODALITY_FEATURES,
    MISSING_COLS,
    STAGE_LABELS,
    TARGET_COL,
    WEARABLE_COLS,
)


@dataclass
class Preprocessor:
    """Simple median imputation and standardization fitted on training data only."""
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Preprocessor":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Training features must be a non-empty 2-D array")
        median = np.zeros(x.shape[1], dtype=np.float64)
        for j in range(x.shape[1]):
            finite = x[np.isfinite(x[:, j]), j]
            median[j] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(x), x, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return cls(median=median, mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.median.shape[0]:
            raise ValueError("Feature dimensions do not match fitted preprocessor")
        filled = np.where(np.isfinite(x), x, self.median)
        return np.clip((filled - self.mean) / self.scale, -10.0, 10.0).astype(np.float32)


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    return {
        "n_test_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=STAGE_LABELS, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=STAGE_LABELS, output_dict=True, zero_division=0
        ),
    }


def build_classical_model(kind: str, seed: int = 42) -> Pipeline:
    if kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", estimator),
        ])
    if kind == "logistic_regression":
        estimator = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
    raise ValueError("kind must be 'random_forest' or 'logistic_regression'")


def benchmark_modalities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str = "random_forest",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit comparable classical models on each feature group and fused inputs."""
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    y_train = train[TARGET_COL].astype(int).to_numpy()
    y_test = test[TARGET_COL].astype(int).to_numpy()
    for group, columns in MODALITY_FEATURES.items():
        model = build_classical_model(kind, seed=seed)
        model.fit(train[columns].astype(float), y_train)
        pred = model.predict(test[columns].astype(float)).astype(int)
        metrics = classification_metrics(y_test, pred)
        rows.append({
            "model": kind.replace("_", " ").title(),
            "feature_group": group,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "n_test_examples": metrics["n_test_examples"],
        })
        predictions[group] = pred
    if kind == "random_forest":
        persistence_pred = test["current_stage"].astype(int).to_numpy()
        persistence = classification_metrics(y_test, persistence_pred)
        rows.append({
            "model": "Persistence baseline",
            "feature_group": "Current stage only",
            "accuracy": persistence["accuracy"],
            "macro_f1": persistence["macro_f1"],
            "n_test_examples": persistence["n_test_examples"],
        })
        predictions["Current stage only"] = persistence_pred
        majority_stage = int(pd.Series(y_train).mode().iloc[0])
        majority_pred = np.full(len(y_test), majority_stage, dtype=int)
        majority = classification_metrics(y_test, majority_pred)
        rows.append({
            "model": f"Majority baseline (stage {majority_stage})",
            "feature_group": "Training majority class",
            "accuracy": majority["accuracy"],
            "macro_f1": majority["macro_f1"],
            "n_test_examples": majority["n_test_examples"],
        })
        predictions["Training majority class"] = majority_pred
    return pd.DataFrame(rows), predictions


class MultimodalFusionNet(nn.Module):
    """Separate MLP encoders for image, wearable and clinical feature groups."""
    def __init__(self, n_stages: int = 4):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(len(IMAGE_COLS), 24), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(24, 16), nn.ReLU(),
        )
        self.wearable_encoder = nn.Sequential(
            nn.Linear(len(WEARABLE_COLS), 20), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(20, 16), nn.ReLU(),
        )
        clinical_size = len(CLINICAL_COLS) + len(MISSING_COLS)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_size, 24), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(24, 16), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.15))
        self.classifier = nn.Linear(32, n_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = self.image_encoder(x[:, 0:8])
        wearable = self.wearable_encoder(x[:, 8:14])
        clinical = self.clinical_encoder(x[:, 14:22])
        fused = self.fusion(torch.cat([image, wearable, clinical], dim=1))
        return self.classifier(fused)


def train_neural_fusion(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int = 20,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Train fusion model; fit preprocessing on train only; select by validation macro-F1."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    preprocessor = Preprocessor.fit(train[FEATURE_COLS].to_numpy(dtype=float))
    # Explicit writable arrays avoid PyTorch warnings when a pandas-backed NumPy
    # view is read-only on a particular pandas/NumPy version.
    x_train = np.array(preprocessor.transform(train[FEATURE_COLS].to_numpy(dtype=float)), dtype=np.float32, copy=True)
    x_val = np.array(preprocessor.transform(validation[FEATURE_COLS].to_numpy(dtype=float)), dtype=np.float32, copy=True)
    x_test = np.array(preprocessor.transform(test[FEATURE_COLS].to_numpy(dtype=float)), dtype=np.float32, copy=True)
    y_train = np.array(train[TARGET_COL].to_numpy(dtype=np.int64), dtype=np.int64, copy=True)
    y_val = np.array(validation[TARGET_COL].to_numpy(dtype=np.int64), dtype=np.int64, copy=True)
    y_test = np.array(test[TARGET_COL].to_numpy(dtype=np.int64), dtype=np.int64, copy=True)

    model = MultimodalFusionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_metrics = classification_metrics(y_val, val_pred)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        })
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(x_test))
        predictions = test_logits.argmax(dim=1).cpu().numpy().astype(int)
    metrics = classification_metrics(y_test, predictions)
    return metrics, predictions, pd.DataFrame(history)


def mask_test_modalities(test: pd.DataFrame, probability: float, seed: int = 42) -> pd.DataFrame:
    """Apply test-time modality dropout without mutating the input frame.

    This is a stress test of a model trained on the original training split. It is not
    equivalent to training a model with modality dropout or to a missingness study.
    """
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    rng = np.random.default_rng(seed)
    masked = test.copy(deep=True)
    image_mask = rng.random(len(masked)) < probability
    wearable_mask = rng.random(len(masked)) < probability
    if image_mask.any():
        masked.loc[image_mask, IMAGE_COLS] = np.nan
        masked.loc[image_mask, "image_missing"] = 1
    if wearable_mask.any():
        masked.loc[wearable_mask, WEARABLE_COLS] = np.nan
        masked.loc[wearable_mask, "wearable_missing"] = 1
    return masked


def evaluate_missingness_stress(
    train: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50),
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate fixed fused random forest under added inference-time modality dropout."""
    model = build_classical_model("random_forest", seed=seed)
    model.fit(train[FEATURE_COLS].astype(float), train[TARGET_COL].astype(int))
    rows: list[dict] = []
    y_test = test[TARGET_COL].astype(int).to_numpy()
    for probability in probabilities:
        masked = mask_test_modalities(test, probability, seed=seed + int(probability * 1000))
        pred = model.predict(masked[FEATURE_COLS].astype(float))
        metrics = classification_metrics(y_test, pred)
        rows.append({
            "additional_test_time_modality_dropout": probability,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "n_test_examples": metrics["n_test_examples"],
        })
    return pd.DataFrame(rows)


def subgroup_metrics(test: pd.DataFrame, predictions: np.ndarray, train: pd.DataFrame) -> pd.DataFrame:
    """Report descriptive test metrics by available age/sex codes, if provided."""
    if len(predictions) != len(test):
        raise ValueError("Prediction count must match test rows")

    age_train = pd.to_numeric(train["age_scaled"], errors="coerce")
    age_test = pd.to_numeric(test["age_scaled"], errors="coerce")
    sex_test = pd.to_numeric(test["sex_binary"], errors="coerce")
    if age_train.notna().any():
        threshold = float(age_train.median())
        age_label = f"Age <= training median ({threshold * 100:.1f} years)"
        age_label_high = f"Age > training median ({threshold * 100:.1f} years)"
        age_low_mask = (age_test <= threshold).fillna(False).to_numpy(dtype=bool)
        age_high_mask = (age_test > threshold).fillna(False).to_numpy(dtype=bool)
    else:
        age_label = "Age <= training median (unavailable)"
        age_label_high = "Age > training median (unavailable)"
        age_low_mask = np.zeros(len(test), dtype=bool)
        age_high_mask = np.zeros(len(test), dtype=bool)

    groups = {
        "Sex/code group 0": sex_test.eq(0).fillna(False).to_numpy(dtype=bool),
        "Sex/code group 1": sex_test.eq(1).fillna(False).to_numpy(dtype=bool),
        age_label: age_low_mask,
        age_label_high: age_high_mask,
    }
    y_true = test[TARGET_COL].astype(int).to_numpy()
    rows = []
    for name, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "subgroup": name,
                "n": 0,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "caution": "No available test examples for this group",
            })
            continue
        metrics = classification_metrics(y_true[mask], np.asarray(predictions)[mask])
        rows.append({
            "subgroup": name,
            "n": n,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "caution": "Small subgroup; descriptive only" if n < 30 else "Descriptive only; not evidence of fairness",
        })
    return pd.DataFrame(rows)
