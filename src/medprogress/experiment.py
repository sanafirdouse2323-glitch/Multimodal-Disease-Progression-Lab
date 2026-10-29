"""Orchestration for the reproducible end-to-end demonstration."""
from __future__ import annotations

from typing import Any
import platform

import numpy as np
import pandas as pd
import sklearn
import torch

from .causal import simulate_treatment_effect
from .data import TARGET_COL, create_forecast_examples, make_synthetic_cohort, split_by_patient
from .evaluation import (
    benchmark_modalities,
    evaluate_missingness_stress,
    subgroup_metrics,
    train_neural_fusion,
)


def run_experiment(
    n_patients: int = 400,
    seed: int = 42,
    epochs: int = 20,
    cohort: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the progression workflow on generated or validated visit-level data.

    The separate causal-effect example always uses a new synthetic dataset; it is
    not estimated from a user-supplied progression cohort.
    """
    is_synthetic = cohort is None
    if cohort is None:
        cohort = make_synthetic_cohort(n_patients=n_patients, seed=seed)
    else:
        cohort = cohort.copy(deep=True)
    examples = create_forecast_examples(cohort)
    splits = split_by_patient(examples, seed=seed)

    training_classes = set(splits.train[TARGET_COL].astype(int).unique())
    if len(training_classes) < 2:
        class_description = (
            ", ".join(str(value) for value in sorted(training_classes))
            if training_classes else "none"
        )
        raise ValueError(
            "The training partition contains only one next-visit stage class "
            f"({class_description}). The classifiers need at least two target classes. "
            "Use more participants and follow-up records with different next-visit stages, "
            "and verify the stage-label mapping before rerunning."
        )

    rf_metrics, rf_predictions = benchmark_modalities(
        splits.train, splits.test, kind="random_forest", seed=seed
    )
    lr_metrics, _ = benchmark_modalities(
        splits.train, splits.test, kind="logistic_regression", seed=seed
    )
    benchmarks = pd.concat([rf_metrics, lr_metrics], ignore_index=True)

    neural_metrics, neural_predictions, history = train_neural_fusion(
        splits.train, splits.validation, splits.test, epochs=epochs, seed=seed
    )
    neural_row = pd.DataFrame([{
        "model": "Neural multimodal fusion",
        "feature_group": "Multimodal fusion",
        "accuracy": neural_metrics["accuracy"],
        "macro_f1": neural_metrics["macro_f1"],
        "n_test_examples": neural_metrics["n_test_examples"],
    }])
    benchmarks = pd.concat([benchmarks, neural_row], ignore_index=True)

    missingness = evaluate_missingness_stress(splits.train, splits.test, seed=seed)
    fairness = subgroup_metrics(splits.test, neural_predictions, splits.train)
    _, causal = simulate_treatment_effect(seed=seed)

    return {
        "data_mode": "synthetic" if is_synthetic else "uploaded_csv",
        "cohort": cohort,
        "examples": examples,
        "splits": splits,
        "benchmarks": benchmarks.sort_values(["model", "feature_group"]).reset_index(drop=True),
        "random_forest_predictions": rf_predictions["Multimodal fusion"],
        "neural_predictions": neural_predictions,
        "neural_metrics": neural_metrics,
        "training_history": history,
        "missingness": missingness,
        "fairness": fairness,
        "causal": causal,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
