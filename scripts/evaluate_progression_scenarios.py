"""Compare progression-signal scenarios across independent synthetic cohorts.

This is a simulation audit, not clinical validation. It reports model performance
relative to the persistence baseline and patient-level bootstrap intervals for the
paired accuracy difference within each run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medprogress.data import TARGET_COL, create_forecast_examples, make_synthetic_cohort, split_by_patient  # noqa: E402
from medprogress.evaluation import benchmark_modalities, train_neural_fusion  # noqa: E402


SCENARIOS = ("no_signal", "progression_signal", "difficult")


def patient_bootstrap_accuracy_delta(
    test: pd.DataFrame,
    predictions: np.ndarray,
    persistence_predictions: np.ndarray,
    *,
    seed: int,
    replicates: int = 500,
) -> tuple[float, float]:
    """Percentile 95% interval for accuracy(model) - accuracy(persistence).

    Resampling is by patient, retaining all test windows for a sampled patient.
    The same sampled patient indices are used for both methods (paired bootstrap).
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    if len(test) != len(predictions) or len(test) != len(persistence_predictions):
        raise ValueError("Test rows and both prediction arrays must have equal lengths")

    truth = test[TARGET_COL].astype(int).to_numpy()
    patient_ids = test["patient_id"].to_numpy()
    grouped_indices = {
        patient_id: np.flatnonzero(patient_ids == patient_id)
        for patient_id in pd.unique(patient_ids)
    }
    unique_patients = np.array(list(grouped_indices), dtype=object)
    if len(unique_patients) < 2:
        raise ValueError("At least two test patients are required for patient-level bootstrapping")

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for rep in range(replicates):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_rows = np.concatenate([grouped_indices[patient_id] for patient_id in sampled_patients])
        deltas[rep] = (
            accuracy_score(truth[sampled_rows], predictions[sampled_rows])
            - accuracy_score(truth[sampled_rows], persistence_predictions[sampled_rows])
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return float(lower), float(upper)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.patients < 30:
        raise ValueError("--patients must be at least 30")
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be at least 1")

    rows: list[dict] = []
    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            cohort = make_synthetic_cohort(
                n_patients=args.patients,
                seed=seed,
                scenario=scenario,
            )
            examples = create_forecast_examples(cohort)
            splits = split_by_patient(examples, seed=seed)
            truth = splits.test[TARGET_COL].astype(int).to_numpy()
            persistence = splits.test["current_stage"].astype(int).to_numpy()
            persistence_accuracy = float(accuracy_score(truth, persistence))

            rf_metrics, rf_predictions = benchmark_modalities(
                splits.train, splits.test, kind="random_forest", seed=seed
            )
            lr_metrics, lr_predictions = benchmark_modalities(
                splits.train, splits.test, kind="logistic_regression", seed=seed
            )
            neural_metrics, neural_predictions, _ = train_neural_fusion(
                splits.train, splits.validation, splits.test, epochs=args.epochs, seed=seed
            )

            model_predictions = {
                "Random Forest (imaging only)": rf_predictions["Imaging"],
                "Random Forest (wearables only)": rf_predictions["Wearables"],
                "Random Forest (clinical/current stage)": rf_predictions["Clinical + temporal/current stage"],
                "Random Forest (multimodal)": rf_predictions["Multimodal fusion"],
                "Logistic Regression (imaging only)": lr_predictions["Imaging"],
                "Logistic Regression (wearables only)": lr_predictions["Wearables"],
                "Logistic Regression (clinical/current stage)": lr_predictions["Clinical + temporal/current stage"],
                "Logistic Regression (multimodal)": lr_predictions["Multimodal fusion"],
                "Neural multimodal fusion": neural_predictions,
            }
            metric_frames = {
                "Random Forest": rf_metrics,
                "Logistic Regression": lr_metrics,
            }
            feature_group_for_model = {
                "imaging only": "Imaging",
                "wearables only": "Wearables",
                "clinical/current stage": "Clinical + temporal/current stage",
                "multimodal": "Multimodal fusion",
            }

            for model_name, predictions in model_predictions.items():
                accuracy = float(accuracy_score(truth, predictions))
                if model_name == "Neural multimodal fusion":
                    macro_f1 = float(neural_metrics["macro_f1"])
                else:
                    estimator_name = model_name.split(" (", maxsplit=1)[0]
                    suffix = model_name.split(" (", maxsplit=1)[1].rstrip(")")
                    group_name = feature_group_for_model[suffix]
                    matching = metric_frames[estimator_name]
                    matching = matching[matching["feature_group"] == group_name]
                    macro_f1 = float(matching.iloc[0]["macro_f1"])

                delta = accuracy - persistence_accuracy
                ci_low, ci_high = patient_bootstrap_accuracy_delta(
                    splits.test,
                    predictions,
                    persistence,
                    seed=seed + 100_000 + len(rows),
                    replicates=args.bootstrap_replicates,
                )
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "patients": int(cohort["patient_id"].nunique()),
                    "test_patients": int(splits.test["patient_id"].nunique()),
                    "test_windows": int(len(splits.test)),
                    "model": model_name,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "persistence_accuracy": persistence_accuracy,
                    "accuracy_delta_vs_persistence": delta,
                    "patient_bootstrap_95ci_delta_low": ci_low,
                    "patient_bootstrap_95ci_delta_high": ci_high,
                    "bootstrap_interval_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                })

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["scenario", "model"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy_across_seeds=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_persistence_accuracy=("persistence_accuracy", "mean"),
            mean_accuracy_delta_vs_persistence=("accuracy_delta_vs_persistence", "mean"),
            fraction_seed_runs_beating_persistence=("accuracy_delta_vs_persistence", lambda x: float((x > 0).mean())),
            seed_delta_p05=("accuracy_delta_vs_persistence", lambda x: float(x.quantile(0.05))),
            seed_delta_p95=("accuracy_delta_vs_persistence", lambda x: float(x.quantile(0.95))),
            bootstrap_intervals_excluding_zero=("bootstrap_interval_excludes_zero", "sum"),
        )
    )
    summary["note"] = (
        "Synthetic scenario only; seed-to-seed spread is not a clinical confidence interval. "
        "Bootstrap interval is for paired accuracy difference versus persistence within each run."
    )
    return runs, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multimodal models with persistence across three synthetic progression scenarios."
    )
    parser.add_argument("--patients", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent seeds per scenario.")
    parser.add_argument("--epochs", type=int, default=5, help="Neural fusion epochs per run.")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--runs-output", type=Path, default=Path("progression_scenario_runs.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("progression_scenario_summary.csv"))
    args = parser.parse_args()

    runs, summary = run(args)
    args.runs_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.runs_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(f"Wrote per-run results to {args.runs_output}")
    print(f"Wrote scenario summary to {args.summary_output}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
