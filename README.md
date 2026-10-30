# Multimodal Disease Progression Lab

**Live interactive demo:** https://multimodal-disease-progression-lab.onrender.com  
**GitHub repository:** https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab

I built this project as an educational machine-learning and research-methods prototype. I use it to demonstrate how a workflow can combine imaging-related numerical features, wearable-style numerical features, and clinical or visit-level information when estimating a participant's stage at their next recorded visit.

The dashboard groups its data-source choices, empty CSV template download, and experiment settings in the centre of the page. The choices are Synthetic demo, Enter simple numbers, and Upload CSV. A first-time visitor can use the pre-filled example or edit a small table without preparing a CSV. The detailed upload option remains available for users who already have numerical features.

The default experience uses entirely synthetic data. I do not claim that the synthetic data represent a real disease, that the generated relationships are medically established, or that the models are ready for clinical use.

> **Important:** This is a research and learning prototype, not a diagnostic, prognostic, or treatment tool. The hosted website is public and receives uploaded files. Do not upload identifiable, confidential, or unauthorised health information. The app does not accept raw MRI, X-ray, image, or wearable sensor files.

## What I built

The project contains an interactive Streamlit dashboard and a Python experiment pipeline. I organised the workflow around five methods-oriented themes:

1. **Multimodal learning:** comparing imaging-related, wearable-style, and clinical/temporal feature groups individually and together.
2. **Longitudinal progression:** using the measurements from one visit to predict the stage label at the next observed visit.
3. **Missing-modality analysis:** examining what happens when an imaging or wearable feature group is additionally masked at evaluation time.
4. **Synthetic causal-effect estimation:** comparing simple estimators when the treatment effect and data-generating assumptions are known by construction.
5. **Descriptive subgroup evaluation:** reporting model metrics for the available age and numeric group-code subgroups.

These themes describe the software demonstrations in this repository. They are not claims that I have established a clinical method or validated a medical model.

## What the application does

I designed the dashboard so that a visitor can start without supplying a dataset.

| Area | What I implemented |
|---|---|
| Synthetic data | Generates artificial participants with multiple visits, stage labels, feature groups, irregular visit gaps, and configurable missingness. |
| Scenario selection | Provides a negative-control scenario, a scenario with a deliberately planted noisy progression signal, and a more difficult low-signal/high-missingness scenario. |
| CSV input | Validates a visit-level table containing participant IDs, visit times, stage labels, and optional numerical features. |
| Progression models | Compares Random Forest, Logistic Regression, a neural multimodal fusion model, and simpler baselines. |
| Modality comparison | Evaluates imaging-related, wearable-style, clinical/temporal, and fused feature groups. |
| Missing-data stress test | Applies additional inference-time modality dropout to a Random Forest trained on the original training split. |
| Causal simulation | Compares unadjusted, adjusted-regression, and inverse-probability-weighted estimates against a known synthetic effect. |
| Subgroup evaluation | Reports descriptive metrics by available age groups and numeric group codes. |
| Exports | Provides benchmark metrics, held-out predictions, an experiment-summary JSON file, and a blank CSV template. |

## Start with a few whole numbers

For visitors who do not work with research datasets, I added a simple number-entry option directly on the homepage.

1. Choose **Enter simple numbers** in the centre of the page.
2. Each row represents one person. Enter the stage number at month 0, month 6, and month 12.
3. Use whole numbers only. The allowed stage labels are 0, 1, 2, and 3; decimal points are not needed.
4. The example starts with 30 artificial rows. You can edit the values, add rows, or download a blank four-column number sheet.
5. Select **Run with my numbers** to run the example.

The stage numbers are generic category labels. They are not diagnoses, disease severity levels, or a validated medical scale. This quick-entry route uses visit times and stage labels only. It does **not** create scan measurements or wearable-device measurements, so it is not a full multimodal experiment. The same four-column sheet can also be uploaded using **Upload CSV**.

The data-source choices, empty CSV template download, and experiment settings are grouped together in the centre of the homepage, so visitors do not need to use a sidebar. After an experiment runs, a plain-language **Start here: how this demo works** guide appears below the results. The detailed CSV option is for users who already have a longitudinal visit table and, where relevant, pre-computed numerical image or wearable features.

The dashboard also provides separate views for prediction results, data-group comparisons, missing-information checks, a practice simulation, group summaries, and data downloads. On the simple-number route, it clearly marks analyses that cannot be run because scan, wearable, or subgroup information was not supplied. The held-out prediction export includes the Random Forest, Logistic Regression, and neural-model predictions.

## How I define the prediction task

Each participant has a sequence of visits. For a given visit, the model receives features recorded at that visit and predicts the stage label recorded at the participant's next observed visit.

For example, if the stage code is 1 at month 6 and the next recorded stage code is 2 at month 12, the month-6 example has target label 2. The time between visits can vary. I therefore describe this as **next-observed-visit classification**, not a fixed 6-month, 12-month, or other fixed-horizon forecast.

The stage labels 0, 1, 2, and 3 are generic codes in this prototype. Their meaning depends on the mapping chosen by whoever supplies a dataset. The software does not define or validate a medical staging system.

I split data by participant, rather than randomly splitting individual visit rows. The default split is 60% training participants, 20% validation participants, and 20% test participants. This keeps visits belonging to one participant in one partition. For the neural model, I fit imputation and standardisation values using the training partition only. I select the neural checkpoint using validation macro-F1; I do not use test labels for training or checkpoint selection.

## The three synthetic scenarios

I added three controlled scenarios so that the software can be examined under different artificial data-generating assumptions.

| Scenario | What I simulate | What the scenario is for |
|---|---|---|
| **progression_signal** (default) | A hidden patient-level tendency influences the probability of an upward stage transition. I encode that tendency noisily in generated imaging-related, wearable-style, and laboratory features. The hidden value itself is not supplied to the models. | Tests whether a model can recover a deliberately planted, noisy relationship. |
| **no_signal** | Stage transitions follow stage-dependent random rules, without a patient-specific progression tendency being reflected in the measurements. | Acts as a negative control. A learned model should not be assumed to outperform persistence. |
| **difficult** | I weaken the planted feature signals, increase measurement noise, and raise imaging and wearable modality missingness. | Tests the pipeline under a deliberately harder synthetic setting. |

These are software test conditions. The planted relationship is an assumption in my generator, not a known medical mechanism. A model's ability to detect a relationship in this simulation does not demonstrate that the relationship exists in real patients.

## Models and baselines

I compare the following approaches:

- **Persistence baseline:** predicts that the next stage will equal the current stage.
- **Majority-class baseline:** predicts the most common next-stage class in the training data.
- **Random Forest:** fits separate models to each feature group and to the combined feature set.
- **Logistic Regression:** provides the same feature-group comparisons using a linear classifier.
- **Neural multimodal fusion:** uses separate small neural-network encoders for imaging-related, wearable-style, and clinical/temporal inputs, then combines their representations for four-class prediction.

I report accuracy and macro-F1. Accuracy measures the fraction of correct predictions. Macro-F1 calculates F1 for each of the four stage labels and averages the class scores equally, so performance on less common labels is not hidden by the largest class.

I also report exact next-stage accuracy on test windows where the stage actually changed, together with precision, recall, and F1 for detecting whether any stage change occurred. I added these measures because an overall score can look high when most next-visit labels stay unchanged.

## What the current synthetic evaluation shows

I ran the repeated-seed evaluator in GitHub Actions run 43 using 400 synthetic participants per run, 10 seeds per scenario, 5 neural training epochs, and 500 patient-bootstrap replicates for each model/run comparison. The following are the mean **overall accuracy** values reported by that run:

| Scenario | Persistence | Random Forest (multimodal) | Logistic Regression (multimodal) | Neural multimodal fusion |
|---|---:|---:|---:|---:|
| progression_signal | 85.28% | 83.19% | 83.93% | 54.56% |
| no_signal | 85.74% | 85.30% | 84.61% | 54.41% |
| difficult | 85.28% | 84.83% | 84.61% | 56.82% |

**My interpretation:** in this evaluation, none of the three multimodal models achieved a higher mean overall accuracy than the persistence baseline. The neural model performed substantially worse on this task in these runs. This is an important limitation, not a result I will hide or describe as a successful predictive model.

The evaluation script also measures performance specifically on changed-stage windows and records paired patient-level bootstrap intervals for the accuracy difference from persistence. Those intervals are simulation diagnostics for this experiment; they are not clinical confidence intervals. The full per-run CSVs created by the CI job are temporary CI outputs and are not committed to this repository. You can reproduce the analysis locally using the command below.

Results can change with the data-generating assumptions, sample size, random seeds, dependency versions, and training settings. I do not present these synthetic results as clinical performance.

## Run the project locally

I recommend Python 3.12, which is the version recorded in the repository's .python-version file.

1. Clone the repository and enter the project directory:

    git clone https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab.git
    cd Multimodal-Disease-Progression-Lab

2. Create and activate a virtual environment.

   Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

   macOS or Linux:

    python -m venv .venv
    source .venv/bin/activate

3. Install the dependencies and launch the dashboard:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    streamlit run app.py

4. Open the local address printed by Streamlit in your browser.

The requirements file installs CPU-only PyTorch. A GPU is not required for this demonstration.

## Run the tests and experiments

From the repository root, run the automated tests:

Windows PowerShell:

    $env:PYTHONPATH = "src"
    python -m pytest -q

macOS or Linux:

    PYTHONPATH=src python -m pytest -q

Run one reproducible synthetic experiment:

    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario progression_signal --output smoke-summary.json

Choose any of the three supported scenario names:

    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario no_signal
    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario difficult

Run the repeated-seed evaluation across all three scenarios:

    python scripts/evaluate_progression_scenarios.py --patients 400 --seeds 10 --epochs 5 --bootstrap-replicates 500 --runs-output progression_scenario_runs.csv --summary-output progression_scenario_summary.csv

The repeated-seed script writes one CSV with per-run results and another with aggregated scenario summaries. It compares Random Forest, Logistic Regression, and neural fusion against persistence, includes modality ablations, records changed-stage metrics, and calculates paired patient-level bootstrap intervals for accuracy differences. These CSV files are generated outputs; I do not treat them as source files, and I do not commit them unless there is a specific reason to preserve a result.

## Use a spreadsheet of my own data

The homepage offers three choices in the centre of the page:

- **Synthetic demo:** runs the built-in artificial example. No file is needed.
- **Enter simple numbers:** lets a visitor edit a compact table with a person number and three stage labels. The blank number sheet uses the same format and can be uploaded again.
- **Upload CSV:** accepts either the simple four-column number sheet or a detailed longitudinal CSV.

For a detailed CSV, the required columns are **patient_id**, **time_months**, and **current_stage**. Optional columns are **age_years**, **sex_binary**, **lab_a**, **lab_b**, **img_0** through **img_7**, and **wear_0** through **wear_5**.

The detailed CSV path accepts numerical features that have already been extracted. It does not read image files or process raw wearable sensor streams. The complete schema, validation rules, and upload instructions are in [UPLOAD_DATA_GUIDE.md](UPLOAD_DATA_GUIDE.md).

The public hosted application receives any file uploaded to it. Do not upload identifiable or confidential health information. For sensitive or regulated research data, use an institutionally approved secure environment and follow the relevant consent, ethics, and data-governance requirements.

A CSV passing format validation only means that it meets the checks implemented by this prototype. It does not establish that the dataset is clinically meaningful, representative, unbiased, legally usable, or suitable for drawing a research conclusion.

## Important methodological limitations

- The next visit may occur after different time intervals; the task is not a fixed-horizon forecast.
- The generated stage labels and relationships are artificial and have no inherent medical interpretation.
- A feature-level CSV upload is not raw-image or raw-sensor analysis.
- The models may not outperform the persistence baseline. The current repeated-seed results show that they do not do so on mean overall accuracy.
- The additional missing-modality test masks modalities at evaluation time after training on the original training split. It does not prove robustness to real-world missing-not-at-random data.
- The causal-effect tab uses its own synthetic dataset and known effect. It does not estimate a causal effect from an uploaded progression CSV.
- The subgroup results are descriptive. They do not establish fairness, and the optional numeric group code is not interpreted as a particular identity or category by the software.
- The prototype has not been clinically validated, externally validated, or approved for patient care.

I describe the implementation details and limitations in [METHODS_AND_LIMITATIONS.md](METHODS_AND_LIMITATIONS.md).

## Repository structure

| Path | Purpose |
|---|---|
| app.py | Streamlit dashboard and visitor instructions. |
| src/medprogress/data.py | Synthetic data generation, CSV validation, forecast-example construction, and participant-level splitting. |
| src/medprogress/evaluation.py | Classical models, neural fusion, metrics, missing-modality stress testing, and subgroup summaries. |
| src/medprogress/causal.py | Separate synthetic treatment-effect simulation with known ground truth. |
| src/medprogress/experiment.py | End-to-end experiment orchestration. |
| src/medprogress/simple_input.py | Integer-only number table for visitors who do not have a detailed CSV. |
| scripts/run_experiment.py | Command-line runner for one synthetic experiment. |
| scripts/evaluate_progression_scenarios.py | Repeated-seed, multi-scenario comparison and bootstrap diagnostics. |
| tests/ | Regression tests for the app, data handling, models, causal simulation, uploads, and synthetic scenarios. |
| .github/workflows/tests.yml | GitHub Actions tests, smoke experiment, and repeated-seed evaluation. |
| examples/sample_run_summary.json | Historical synthetic output recorded before the three-scenario generator was introduced. |
| templates/longitudinal_upload_template.csv | Blank longitudinal CSV template. |
| UPLOAD_DATA_GUIDE.md | Detailed CSV schema, validation and privacy instructions. |
| METHODS_AND_LIMITATIONS.md | Explanation of the workflow, model design, evaluation, and scientific boundaries. |
| GITHUB_SETUP.md | Local setup and GitHub contribution instructions. |
| DEPLOYMENT.md | Render deployment configuration and verification steps. |

## How I maintain the project

I use GitHub Actions to run the tests and synthetic experiments on pull requests and pushes to main. My Render service is connected to the main branch with automatic deployment enabled. A successful GitHub merge does not, by itself, prove that the website has deployed successfully; I check the Render deployment record and confirm the deployed commit before describing a revision as live.

For maintenance instructions, see [GITHUB_SETUP.md](GITHUB_SETUP.md) and [DEPLOYMENT.md](DEPLOYMENT.md).

---

I built this repository as a transparent, reproducible educational prototype. I welcome careful inspection of its code and results, especially where the evidence shows that the current models still need improvement.
