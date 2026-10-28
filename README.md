# Multimodal Disease Progression Lab

A runnable, educational machine-learning-in-medicine research demonstration aligned with five broad themes in a multimodal disease-progression PhD advertisement: multimodal fusion, sparse longitudinal progression, causal treatment-effect estimation, missing modalities, and subgroup evaluation.

**Synthetic data only.** This project does not use real medical images or patient records, diagnose disease, recommend treatment, or establish clinical performance. Stage labels and all simulated outcomes are artificial. The project is a software/methods demonstration, not a clinical tool or a claim to reproduce any supervisor's unpublished research.

## What is implemented

- Reproducible synthetic patients with irregularly spaced visits and image-derived, wearable and tabular feature groups.
- Next-observed-visit stage classification with patient-level train/validation/test splits.
- Training-only preprocessing for neural inputs and preprocessing pipelines for classical models.
- Logistic regression, random forest, persistence and majority-class baselines.
- A PyTorch fusion model with separate encoders for each modality.
- Test-time missing-modality stress tests, a controlled causal simulation with known ground truth, and descriptive subgroup metrics.
- Streamlit UI, CSV/JSON exports, automated tests, and GitHub Actions CI.

## Run locally

```bash
git clone https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab.git
cd Multimodal-Disease-Progression-Lab
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`; on macOS/Linux, use `source .venv/bin/activate` after creating the virtual environment.

## Verify and run headless

```bash
PYTHONPATH=src python -m pytest -q
python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --output /tmp/smoke-summary.json
```

## Evaluation cautions

The target is the stage at the next observed visit, not a fixed 6- or 12-month forecast. The model feature list excludes patient IDs, future visit time, latent synthetic risk, and the separate `synthetic_age`/`synthetic_sex` metadata columns. Age and sex are deliberately included as tabular demo inputs (`age_scaled` and `sex_binary`); the separate metadata columns support descriptive subgroup analysis. Patients are split before model fitting, and neural preprocessing statistics are learned from training data only. Baselines are included because a complex model is not guaranteed to outperform carrying forward the current stage.

The causal analysis only demonstrates whether estimators recover an effect in a known synthetic data-generating process; it does not establish identification in real observational medicine. Subgroup measures are descriptive and use synthetic demographics. The missing-modality experiment adds test-time modality dropout to a model trained on the original data; it is not proof of robustness to missing-not-at-random data or a substitute for training-time robustness experiments.

## Repository contents

- `app.py` — interactive Streamlit dashboard.
- `src/medprogress/data.py` — synthetic cohort, forecast examples and splits.
- `src/medprogress/evaluation.py` — baselines, neural fusion, metrics and stress tests.
- `src/medprogress/causal.py` — known-ground-truth treatment-effect simulation.
- `src/medprogress/experiment.py` — experiment orchestration.
- `scripts/run_experiment.py` — headless reproducible experiment.
- `tests/` — automated regression tests.
- `.github/workflows/tests.yml` — CI test and smoke experiment.
- `examples/sample_run_summary.json` — saved sample output from a synthetic run.

See `GITHUB_SETUP.md` for maintenance instructions and `DEPLOYMENT.md` for Streamlit Community Cloud deployment. The hosted URL is only assigned after you complete deployment in your authenticated Streamlit workspace. Never commit real patient data, credentials or identifiable health information.
