import numpy as np
import torch

from medprogress.data import FEATURE_COLS, IMAGE_COLS, create_forecast_examples, make_synthetic_cohort, split_by_patient
from medprogress.evaluation import (
    MultimodalFusionNet,
    Preprocessor,
    benchmark_modalities,
    mask_test_modalities,
    subgroup_metrics,
    train_neural_fusion,
    classification_metrics,
)


def _splits(n=80, seed=3):
    examples = create_forecast_examples(make_synthetic_cohort(n, seed=seed))
    return split_by_patient(examples, seed=seed)


def test_classification_metrics_include_balanced_and_class_level_diagnostics():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 2])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["n_test_examples"] == 8
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
    ]
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 8
    assert all(str(stage) in metrics["per_class"] for stage in range(4))


def test_classification_metrics_reject_empty_or_misaligned_inputs():
    for true, pred in (([], []), ([0, 1], [0])):
        try:
            classification_metrics(true, pred)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid labels should raise ValueError")


def test_preprocessor_is_fitted_from_training_values_only():
    train = np.array([[1.0, np.nan], [3.0, 4.0]])
    processor = Preprocessor.fit(train)
    assert processor.median.tolist() == [2.0, 4.0]
    transformed = processor.transform(np.array([[100.0, np.nan]]))
    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()
    assert processor.median.tolist() == [2.0, 4.0]


def test_neural_fusion_forward_has_four_stage_logits():
    model = MultimodalFusionNet()
    output = model(torch.zeros(5, len(FEATURE_COLS)))
    assert tuple(output.shape) == (5, 4)


def test_modality_masking_is_validated_and_does_not_mutate_original():
    splits = _splits(60, 5)
    original = splits.test.copy(deep=True)
    masked = mask_test_modalities(splits.test, probability=1.0, seed=42)
    assert masked["image_missing"].eq(1).all()
    assert masked["wearable_missing"].eq(1).all()
    assert masked[IMAGE_COLS].isna().all().all()
    assert splits.test.equals(original)
    try:
        mask_test_modalities(splits.test, probability=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid probability was not rejected")


def test_classical_benchmarks_and_neural_training_run_end_to_end():
    splits = _splits(90, 6)
    metrics, predictions = benchmark_modalities(splits.train, splits.test, "random_forest", seed=6)
    assert {"Imaging", "Wearables", "Clinical + temporal/current stage", "Multimodal fusion"}.issubset(
        set(metrics.feature_group)
    )
    assert {
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"
    }.issubset(metrics.columns)
    assert np.isfinite(metrics[[
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"
    ]].to_numpy()).all()
    assert len(predictions["Multimodal fusion"]) == len(splits.test)
    neural_metrics, neural_pred, history = train_neural_fusion(
        splits.train, splits.validation, splits.test, epochs=2, seed=6
    )
    assert len(neural_pred) == len(splits.test)
    assert 0 <= neural_metrics["accuracy"] <= 1
    assert 0 <= neural_metrics["macro_f1"] <= 1
    assert len(history) == 2
    subgroup = subgroup_metrics(splits.test, neural_pred, splits.train)
    assert len(subgroup) == 4
