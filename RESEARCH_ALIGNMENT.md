# University of Bern PhD Project Alignment Audit

**Audit reference:** repository `main` at commit `e63d0c5e00473c128b3f43a00a64760e2c484916`  
**Research reference:** the University of Bern Machine Learning in Medicine PhD advertisement and the MLM Lab research background supplied for this project on 31 October 2026.

This document distinguishes what the repository currently implements from what the advertised PhD project proposes. It is an implementation audit, not a claim of clinical validation or a claim that the repository reproduces the lab's research.

## 1. What the advertised PhD project is about

The advertisement describes a research project on learning from multimodal health data to characterise and model disease progression. The stated modalities are medical imaging, wearable sensor data, and clinical measurements. The advertisement lists possible directions rather than requiring every candidate to implement all of them:

- Multimodal representation learning across imaging, time series, and tabular data.
- Disease-progression modelling with cross-sectional and sparse longitudinal data.
- Causal machine learning to disentangle disease progression from treatment effects.
- Learning with missing modalities and irregularly sampled signals.
- Fairness and bias in multimodal models across demographic subgroups.

The lab's broader methodological interests also include uncertainty quantification, robustness and reliability, probabilistic modelling, causal machine learning, interpretable machine learning, performance generalisation and prediction, subgroup discovery, bias detection and mitigation, disentangled representation learning, and simulation-based inference. These are broader lab interests, not all mandatory requirements of this specific PhD project.

## 2. Current repository: verified implementation

The current application is an educational, feature-level prototype. Its code and documentation support the following description.

| Research area | What the repository currently implements | Status relative to the advertisement |
|---|---|---|
| Multimodal data | Compares eight imaging-related numerical columns, six wearable-style numerical columns, and clinical/visit-level numerical features. The synthetic generator creates these values. Uploaded files must contain pre-computed numerical features. | **Partial.** It combines modality-specific feature groups, but it does not process raw medical images or raw wearable streams. |
| Representation learning | Uses separate small multilayer perceptron encoders for the image-related, wearable-style, and clinical/missingness feature groups, then concatenates their representations. | **Basic prototype.** This is a simple learned fusion architecture, not a specialised imaging encoder, time-series encoder, or demonstrated state-of-the-art representation-learning method. |
| Progression modelling | Constructs examples from a participant's current visit and predicts the stage code at that participant's next recorded visit. The stage codes are generic integers 0–3. | **Partial.** This is next-observed-visit classification. It is not yet a continuous trajectory model, time-to-event model, or clinically defined progression endpoint. |
| Sparse/irregular longitudinal data | Accepts multiple visits per participant, requires at least two visits, sorts visits by time, rejects duplicate visit times, and includes the previous visit gap as a feature. | **Partial.** Irregular gaps are represented, but a continuous-time or explicitly horizon-conditioned progression model is not implemented. Cross-sectional-only input is not supported by the current validator. |
| Missing modalities | Generates modality-level missingness indicators and has a test-time stress test that masks image-related or wearable-style features for a fixed Random Forest trained on the original training data. | **Partial.** The current test is inference-time masking. It is not a full study of missing-not-at-random mechanisms and does not train or compare dedicated missing-modality fusion strategies. |
| Causal machine learning | Includes a separate synthetic treatment-effect exercise with a known simulated effect, an unadjusted comparison, adjusted linear regression, and inverse-probability weighting. | **Not yet integrated with progression.** The causal exercise uses its own artificial dataset. Treatment exposure or treatment changes are not modelled in the uploaded progression cohort, so this does not yet disentangle disease progression from treatment effects in longitudinal health data. |
| Fairness and subgroup analysis | Reports descriptive test metrics for two numeric `sex_binary` codes and age groups split at the training median, when those values are supplied. | **Exploratory only.** The current outputs are not a formal fairness or bias audit; they do not establish equitable performance across clinically justified or intersecting subgroups. |
| Uncertainty and reliability | Repeats synthetic experiments across seeds and computes paired patient-level bootstrap intervals for accuracy differences from persistence in the scenario evaluator. | **Partial.** These intervals describe variability in a model-versus-baseline performance comparison. The application does not currently provide calibrated individual prediction probabilities, predictive intervals, or a complete uncertainty-quantification analysis. |
| Interpretability and subgroup discovery | Reports feature-group comparisons and descriptive subgroup metrics. | **Limited.** It does not currently implement a dedicated interpretable model, counterfactual explanation, or subgroup-discovery method. |
| Real-world evidence | Uses artificial data by default and accepts a documented CSV of already prepared numerical features. | **Not demonstrated.** No real clinical dataset, external cohort, raw-image pipeline, or raw wearable-signal pipeline is included. |

## 3. Methodological strengths already present

The following are genuine strengths of the current implementation and are supported by the code:

1. **Participant-level data splitting.** Forecast windows are created first, and participant IDs are then allocated to disjoint training, validation, and test partitions. Visits from the same participant are not randomly scattered across those partitions.
2. **Training-only preprocessing for the neural model.** Its median imputation values, means, and scales are fitted on the training partition and then applied to validation and test data.
3. **Validation-based neural checkpoint selection.** The neural checkpoint is selected using validation macro-F1; the test partition is used for final evaluation.
4. **Reference baselines.** The project includes persistence and training-majority baselines, which are important because next-visit stage labels can remain unchanged.
5. **Multiple feature-group comparisons.** Random Forest and Logistic Regression are run on imaging-related, wearable-style, clinical/temporal, and combined feature groups.
6. **Input validation and provenance.** The CSV path checks required columns, stage-code values, visit times, participant counts, and duplicate visit times; it preserves participant identifiers as text and records whether a cohort came from the synthetic generator or an upload.
7. **Explicit scientific limitations.** The README and methods note state that the stage labels are generic, the uploaded modalities are pre-computed features, the causal exercise is separate, and the prototype is not clinically validated.

These controls make the current project more inspectable than a dashboard that only displays attractive scores. They do not by themselves establish that the predictive models are effective.

## 4. Results that must not be overstated

The README records a repeated-seed synthetic evaluation in which persistence had higher mean overall accuracy than the multimodal Random Forest, Logistic Regression, and neural fusion model in all three listed scenarios. In that recorded evaluation:

- `progression_signal`: persistence 85.28%, Random Forest 83.19%, Logistic Regression 83.93%, neural fusion 54.56%.
- `no_signal`: persistence 85.74%, Random Forest 85.30%, Logistic Regression 84.61%, neural fusion 54.41%.
- `difficult`: persistence 85.28%, Random Forest 84.83%, Logistic Regression 84.61%, neural fusion 56.82%.

These values are recorded synthetic results, not clinical performance estimates and not guaranteed results for every environment. They show that the current models do not beat persistence on mean overall accuracy in that evaluation. That finding should remain visible in the application and in any application discussion.

The scenario sanity test that compares a binary upward-transition AUC in the planted-signal and negative-control generators is a test of the synthetic generator's learnability. It is not evidence that the dashboard models predict real disease progression.

## 5. Highest-priority technical gaps

The next development should address the project's most important scientific gaps before expanding the user interface.

### Priority 1 — Evaluate progression as an event, not only as a four-class label

Keep next-observed-visit stage classification, but add a clearly defined progression endpoint (for example, upward stage transition) and report probability-based discrimination and proper scoring metrics. The endpoint must be specified before evaluation. Results should be reported alongside persistence and class-frequency baselines. A metric should not be selected after looking at test results.

### Priority 2 — Add a defensible uncertainty and calibration evaluation

Expose class probabilities from the models and assess their quality on held-out participants using appropriate multiclass proper scores and reliability diagnostics. Any calibration model must be fitted using training/validation data only, never the test labels. Predictive entropy or maximum softmax probability alone must not be presented as calibrated uncertainty. Report uncertainty limitations and patient-level sampling when estimating intervals.

### Priority 3 — Make missing-modality evaluation more systematic

Distinguish naturally missing inputs from simulated masking. Define and test separate missingness mechanisms (for example, random missingness and missingness dependent on observed variables). Compare test-time masking with methods trained to handle missing modalities. Keep the mechanism and assumptions explicit; do not label a simulated masking experiment as proof of clinical robustness.

### Priority 4 — Improve time-aware progression modelling

Retain each participant's visit times and explicitly assess whether predictions are intended for the next observed visit or a fixed prediction horizon. Compare the current previous-gap feature with a time-aware baseline. Evaluate performance by follow-up interval. Add cross-sectional data support only with a separately defined task and target; a cross-sectional row cannot be treated as a longitudinal transition without a valid outcome definition.

### Priority 5 — Integrate treatment information before making causal claims

The current causal tab is a separate teaching simulation. A progression-and-treatment study would need longitudinal treatment exposure, a clearly specified causal estimand, time ordering, confounder assumptions, and simulation tests that can detect known treatment effects. Do not claim that the current standalone toy simulation removes treatment effects from progression.

### Priority 6 — Strengthen subgroup evaluation

Prespecify clinically justified groups, show subgroup sample sizes and uncertainty, examine relevant intersectional groups when sample sizes permit, and avoid interpreting a numeric code as a demographic identity unless its meaning is documented. Descriptive metrics alone should not be called a fairness guarantee.

### Priority 7 — Add real modalities only when data and governance permit

The current upload format is for pre-computed numerical features. A genuine raw imaging or wearable pipeline would need its own modality-specific preprocessing, quality checks, missingness handling, and provenance. Real health data should only be used in an institutionally approved environment with appropriate permissions, ethics, consent, and data governance. No identifiable or confidential data should be uploaded to the public demo.

## 6. Recommended development order

A careful sequence is:

1. Preserve the current baseline and reproducibility checks; add a documented progression-event endpoint and probability-based evaluation.
2. Add validation-only probability calibration and held-out reliability reporting.
3. Expand missing-modality experiments to include training-time handling and explicitly defined missingness mechanisms.
4. Add interval-aware analyses and a separately designed cross-sectional task only if the target can be justified.
5. Integrate treatment variables and a defined causal estimand before describing treatment-adjusted progression.
6. Expand subgroup analysis and modality-specific processing when data, sample sizes, and governance support them.

Each step should have unit tests, a reproducible synthetic experiment, a clear comparison with the existing baselines, and documentation of what the result does not prove. A change should be merged only after the tests and experiment checks pass.

## 7. Accurate description for a PhD application

A defensible description of the current repository is:

> I developed an educational Python prototype to investigate a longitudinal, feature-level machine-learning workflow using synthetic multimodal data. It compares imaging-related, wearable-style, and clinical feature groups; constructs participant-level next-visit prediction tasks; implements classical and neural fusion baselines; and includes controlled missing-modality, subgroup, and separate synthetic causal-effect exercises. I introduced participant-level data splitting, training-only preprocessing, reproducible scenarios, and explicit baseline comparisons. The current models do not outperform the persistence baseline on mean overall accuracy in the recorded repeated-seed evaluation, so I treat the project as a methodological prototype and a basis for further investigation rather than a validated clinical prediction system.

This wording describes implemented work and preserves the negative result. It should only be used if it accurately reflects the applicant's own contribution and understanding.

---

## Audit boundary

This audit is based on the repository files at the commit identified at the top and on the University of Bern / MLM Lab research material supplied for this project. It does not establish clinical validity, novelty, publication readiness, access to real datasets, or the likely assessment of the application by the selection committee.
