# Multimodal Disease Progression Lab

**Live interactive demo:** https://multimodal-disease-progression-lab.onrender.com

**GitHub source:** https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab


A runnable, educational machine-learning-in-medicine research demonstration aligned with five broad themes in a multimodal disease-progression PhD advertisement: multimodal fusion, sparse longitudinal progression, causal treatment-effect estimation, missing modalities, and subgroup evaluation.

**Default demo: synthetic data.** The app also lets you upload a CSV of tabular, numerical feature-level longitudinal data or download an empty CSV template. Uploaded data are used for progression-model evaluation; the causal-effect tab remains a separate synthetic simulation. The upload option does not accept raw medical images or raw wearable streams.

**Privacy and validity:** The public Render app receives uploaded files. Do not upload identifiable, confidential, or unauthorised health data. Format validation does not establish clinical validity. This is an educational software/methods demonstration, not a clinical tool or a claim to reproduce any supervisor's unpublished research.

## What is implemented

- Reproducible synthetic patients with irregularly spaced visits and imaging-related, wearable-style and tabular feature groups.
- Three explicit synthetic scenarios: no additional progression signal, a planted noisy progression signal, and a difficult low-signal/high-missingness setting.
- Optional CSV upload for a validated visit-level feature table, plus a downloadable blank CSV shell; see `UPLOAD_DATA_GUIDE.md`.
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
python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario progression_signal --output /tmp/smoke-summary.json
```

Run a repeated-seed comparison across all three scenarios (this takes longer because it trains three model families for each seed):

```bash
python scripts/evaluate_progression_scenarios.py --patients 400 --seeds 10 --epochs 5 --bootstrap-replicates 500
```

The script writes per-run results to progression_scenario_runs.csv and aggregated scenario summaries to progression_scenario_summary.csv. It compares Random Forest, Logistic Regression and neural multimodal fusion with persistence, and computes paired patient-level bootstrap intervals for accuracy differences within each run. The seed-to-seed spread and bootstrap intervals are descriptive simulation diagnostics, not clinical confidence intervals.

## Evaluation cautions

The target is the stage at the next observed visit, not a fixed 6- or 12-month forecast. Uploaded CSVs require `patient_id`, numeric `time_months`, and `current_stage` encoded as 0, 1, 2, or 3. Optional columns are `age_years`, `sex_binary`, `lab_a`, `lab_b`, `img_0`–`img_7`, and `wear_0`–`wear_5`. See `UPLOAD_DATA_GUIDE.md` for the complete schema. Extra columns are ignored and reported. Patient IDs are excluded from model features, and splits are made by patient to keep visits for one patient in a single partition. Neural preprocessing statistics are learned from training data only. Baselines are included because a complex model is not guaranteed to outperform carrying forward the current stage.

### Synthetic progression scenarios

- no_signal: transitions use stage-dependent random rules, while the measured modalities have no direct patient-specific progression signal. This is a negative-control scenario; persistence may perform as well as or better than learned models.
- progression_signal (default): a hidden patient-level progression tendency affects upward-transition probability and is reflected noisily in generated image-related, wearable-style and laboratory measurements. This is intentionally planted signal, not an established medical relationship.
- difficult: the planted signal is weaker, measurements are noisier, and imaging/wearable modality missingness is higher.

These scenarios test whether the software can recover an assumed relationship under controlled synthetic conditions. They do not establish that real clinical measurements contain the same relationship. The generator's assumptions should be treated as experimental controls, not medical facts. A model that beats persistence in the signal scenario must still be tested across seeds, checked for uncertainty, and evaluated on appropriately governed real-world data before making any clinical claims.

The causal analysis only demonstrates whether estimators recover an effect in a known synthetic data-generating process; it does not establish identification in real observational medicine. Subgroup measures are descriptive: the synthetic mode uses generated demographics, while uploaded-data mode uses supplied age and sex/code values where available. These metrics do not establish fairness. The missing-modality experiment adds test-time modality dropout to a model trained on the original data; it is not proof of robustness to missing-not-at-random data or a substitute for training-time robustness experiments.

The saved examples/sample_run_summary.json is a historical snapshot from before the three-scenario generator was introduced; its performance numbers are not results from the current default scenario. Regenerate experiments using the commands above for current comparisons. The saved example also records the software versions from its original run, which may differ from the current CI/deployment environment.

## Repository contents

- `app.py` — interactive Streamlit dashboard.
- `src/medprogress/data.py` — synthetic cohort, forecast examples and splits.
- `src/medprogress/evaluation.py` — baselines, neural fusion, metrics and stress tests.
- `src/medprogress/causal.py` — known-ground-truth treatment-effect simulation.
- `src/medprogress/experiment.py` — experiment orchestration.
- `scripts/run_experiment.py` — headless reproducible experiment with scenario selection.
- `scripts/evaluate_progression_scenarios.py` — repeated-seed comparison of multimodal models against persistence, including patient-level bootstrap intervals.
- `tests/` — automated regression tests, including checks for the planted synthetic progression signal.
- `.github/workflows/tests.yml` — CI test and smoke experiment.
- `examples/sample_run_summary.json` — saved sample output from a synthetic run.
- `templates/longitudinal_upload_template.csv` — blank, header-only CSV shell.
- `UPLOAD_DATA_GUIDE.md` — accepted input schema, workflow, and privacy limits.

See `GITHUB_SETUP.md` for maintenance instructions, `DEPLOYMENT.md` for Render configuration, and `UPLOAD_DATA_GUIDE.md` for CSV input rules. Render's free service may sleep when idle. Never commit real patient data, credentials or identifiable health information.
