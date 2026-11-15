# Methods, Model Design, and Limitations

I wrote this note to explain what the current code implements, what assumptions I deliberately introduced into the synthetic generator, and what I can and cannot conclude from the outputs.

## 1. Purpose and scope

I built the Multimodal Disease Progression Lab as an educational software prototype for exploring a longitudinal, feature-level machine-learning workflow. It combines three broad feature groups:

- Imaging-related numerical features.
- Wearable-style numerical features.
- Clinical, demographic, and visit-timing features.

The application does not read raw images, extract features from scans, ingest raw sensor streams, or connect to a clinical information system. For uploaded data, the user must supply numerical features that have already been computed.

I use the term multimodal because the experiment can compare and combine these feature groups. That term does not imply that the current prototype performs end-to-end image or wearable-signal processing.

## 2. Prediction target

For participant i, let the visits be ordered by their observed time. For a visit t, the input consists of numerical information available at that visit. The target is the stage code at the next recorded visit:

    current-visit features  ->  next-observed-visit stage label

The target is one of four generic integer labels: 0, 1, 2, or 3. The code does not attach a clinical definition to those labels. A data owner must define and document any mapping before using a supplied dataset.

Because visit gaps are not fixed, the task is not a fixed-horizon forecast. It predicts the next recorded label, even if different examples have different time intervals. The next-visit interval is recorded for interpretation and export, but it is not a model input. The feature visit_gap_scaled represents the interval since the previous observed visit.

## 3. Synthetic data design

I provide three generator scenarios. All are artificial and reproducible from a selected random seed.

### progression_signal

I generate a hidden patient-level progression tendency. In this scenario, it changes the probability of an upward stage transition. I also encode that tendency noisily in the generated imaging-related, wearable-style, and laboratory measurements. The hidden tendency is not itself included in the model feature table.

### no_signal

I retain stage-dependent random transition rules but remove the link between a patient-specific progression tendency and the observed measurements. This is a negative-control scenario for the planted progression relationship. Measurements can still reflect the current stage and other simulated patient characteristics; the label no_signal specifically means that the additional patient-level progression tendency is not linked to the measurements.

### difficult

I retain a weaker planted progression relationship, increase feature noise, and raise the probability of missing imaging and wearable feature groups. This makes the synthetic signal harder to measure.

These scenarios are experimental controls, not medical hypotheses. Their relationships, scales, feature distributions, and stage transition probabilities are chosen by the generator. They should not be described as disease biology or as evidence about patients.

## 4. Feature groups and preprocessing

The current feature table contains:

- Eight imaging-related numerical columns: img_0 through img_7.
- Six wearable-style numerical columns: wear_0 through wear_5.
- Clinical and visit information: scaled age, a numeric group code, two generic lab fields, scaled current stage, and scaled visit gap.
- Two indicators showing whether the imaging and wearable groups are missing.

For classical models, the code compares four feature groups: imaging, wearables, clinical plus temporal/current-stage information, and all features combined. Random Forest uses median imputation. Logistic Regression uses median imputation followed by standardisation. These preprocessing steps are fitted within each model's training pipeline.

The Random Forest uses 180 trees, a minimum leaf size of 2, and balanced-subsample class weights. Logistic Regression uses balanced class weights and a maximum of 1,200 iterations. The neural fusion model has separate small multilayer-perceptron encoders for imaging, wearable, and clinical/missingness inputs. It concatenates the three encoded representations and passes them through a fusion layer and a four-class output layer. Neural preprocessing is fitted on training data only. The network uses Adam with a learning rate of 0.001, weight decay of 0.0001, batches of 64, and ordinary unweighted cross-entropy loss. It selects the saved checkpoint using validation macro-F1.

The three model types therefore do not use identical class-weighting strategies: both classical classifiers apply class weighting, while the neural model does not. Their scores compare the implemented pipelines; they should not be interpreted as a controlled comparison of architecture alone.

These are compact demonstration models. I do not describe them as a state-of-the-art architecture or as a clinically optimised system.

## 5. Data splitting and leakage controls

I create forecast examples from each participant's consecutive visits. I then make a seeded random split of unique participant IDs into training, validation, and test groups in a 60/20/20 proportion. The split is not stratified by target class. All visits from one participant remain within one partition. The code checks that the training partition contains at least two target classes, but it does not guarantee that all four classes appear in validation or test. Class support can therefore vary, especially in small or imbalanced datasets.

For the neural model, the imputation medians, means, and scales are computed on training features and then applied to validation and test features. The validation partition is used for neural checkpoint selection. The test partition is held for final evaluation.

These choices reduce a common source of leakage from putting visits belonging to the same participant into different partitions. They do not eliminate every possible source of bias or make the evaluation externally valid.

## 6. Models and reference baselines

The application reports:

- **Persistence:** the next stage is predicted to equal the current stage.
- **Majority class:** the most frequent next-stage label in the training set is predicted for every test row.
- **Random Forest:** trained separately on each feature group and on all features combined.
- **Logistic Regression:** trained on the same feature-group definitions.
- **Neural multimodal fusion:** a three-encoder neural network trained on the combined feature representation.

I include the persistence baseline because consecutive stage labels may remain unchanged. A model can achieve apparently high overall accuracy by predicting the current stage even when it does not identify actual changes. For that reason, the dashboard also calculates exact next-stage accuracy among test examples where the stage changed, plus precision, recall, and F1 for the binary task of detecting any change.

## 7. Evaluation metrics

For each held-out classification result, I report:

- **Accuracy:** the fraction of test predictions that are correct.
- **Balanced accuracy:** the average recall across the classes represented in the test labels. Unlike ordinary accuracy, this gives each represented class equal weight.
- **Macro precision, macro recall, and macro-F1:** each metric is calculated across the four defined stage labels with equal class weighting. Macro-F1 is not the same as balanced accuracy. When all four stage labels occur in the test set, balanced accuracy and fixed-label macro recall will generally be equal; if a stage is absent from the test set, balanced accuracy averages the represented classes while the fixed-label macro report still includes all four labels.
- **Per-class precision, recall, F1, and support:** these show the performance and number of test examples for each stage label.
- **Confusion matrix:** rows are actual stage labels and columns are predicted stage labels; counts are calculated on held-out test examples.

The dashboard displays the aggregate measures in the model-comparison table and provides per-class metrics and confusion matrices for the Random Forest, Logistic Regression, and neural multimodal fusion models. The experiment-summary JSON also includes the detailed metrics. A held-out error-analysis table and CSV identify test examples where at least one of those classifiers disagrees with the target label; these are prediction errors for inspection, not a clinical error review. A class absent from the test set has zero support in the fixed-label per-class report; its per-class precision, recall, and F1 require cautious interpretation.

The repeated-seed script additionally reports:

- Performance on windows where the next stage differs from the current stage.
- Precision, recall, and F1 for detecting any stage change.
- Accuracy differences from persistence.
- Paired patient-level bootstrap percentile intervals for those accuracy differences within each run.
- The proportion of seed runs in which each model beats persistence.
- Summary variation across random seeds for accuracy, balanced accuracy, macro precision, macro recall, and macro-F1.

The bootstrap resamples test participants and retains all test windows belonging to each sampled participant. The same sampled participants are used for the model and persistence predictions, making the comparison paired. These are simulation diagnostics for the generated experiment. They are not clinical confidence intervals and should not be used as such.

## 8. Recorded repeated-seed results

The table below records the mean overall accuracy from GitHub Actions run 43 on 29 October 2026. That run used 400 synthetic participants per run, 10 seeds per scenario, 5 neural epochs, and 500 patient-bootstrap replicates. These are the values recorded in the project documentation; the run's generated CSVs were not saved as workflow artifacts, so the exact per-run results cannot now be independently recalculated from that run.

| Scenario | Persistence | Random Forest | Logistic Regression | Neural fusion |
|---|---:|---:|---:|---:|
| progression_signal | 85.28% | 83.19% | 83.93% | 54.56% |
| no_signal | 85.74% | 85.30% | 84.61% | 54.41% |
| difficult | 85.28% | 84.83% | 84.61% | 56.82% |

In this run, the multimodal models did not beat persistence on mean overall accuracy in any scenario. The results show that planting a relationship in the data generator does not automatically mean that the current modelling pipeline learns a useful next-stage classifier. This remains an implementation and modelling problem to investigate.

The values above are a dated synthetic result, not a clinical estimate and not a guarantee for other environments. Later workflow runs 85, 86, and 88 passed their configured tests and repeated-seed commands, but no result artifacts were retained from those runs. The updated PR #10 workflow has saved per-run and summary CSVs for successful runs 89–93. The latest is the [progression-scenario-results-93 artifact](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab/actions/runs/43664302014), which GitHub lists as expiring on 15 December 2026. These artifacts belong to PR-branch runs; the workflow change remains on the PR branch, not main. Running the script locally also creates both CSVs at the paths selected by its command-line options.

## 9. Missing-modality experiment

The application trains a Random Forest on the original training partition. It then applies additional whole-modality masking to the test partition at probabilities 0.00, 0.10, 0.25, and 0.50, updating the corresponding missingness indicators. The model is not retrained for each masking probability.

This is an inference-time stress test of a fixed model. It is not a study of missing-not-at-random mechanisms, it does not compare all possible failure patterns, and it does not prove clinical robustness.

## 10. Separate synthetic causal-effect exercise

The causal tab calls a separate generator with 5,000 simulated individuals by default and a known treatment effect of -0.8 outcome units. Treatment assignment depends on simulated risk and age, creating confounding. The app compares the known simulated effect with:

- The unadjusted treated-minus-control outcome difference.
- An adjusted linear regression coefficient.
- An inverse-probability-weighted difference using an estimated propensity score.

This exercise is kept separate from the progression cohort. Even when a user uploads a CSV, the causal tab does not estimate an effect from that uploaded data. The simulation demonstrates estimator behaviour under its own known data-generating assumptions; it does not identify treatment effects in real observational healthcare data.

## 11. Descriptive subgroup summaries

The app reports descriptive metrics for numeric group codes 0 and 1 and for age groups split at the training-set median, when those values are available. The software does not infer what the numeric group codes represent. Missing values can leave a group empty.

These summaries are not a fairness audit. A proper fairness assessment would require a clearly justified subgroup definition, suitable sample sizes, uncertainty estimates, clinically relevant metrics, analysis of measurement and label quality, and external validation.

## 12. Input checks and privacy

The CSV validator checks the required fields, numeric formats, stage-code range, non-negative visit times, repeated visits, participant counts, and duplicate visit times. It also checks the documented ranges for optional age and group-code fields.

Validation only confirms that the file meets the implemented structural checks. It does not establish clinical suitability, data rights, consent, ethics approval, representativeness, or scientific validity.

The public hosted app receives uploaded files. Do not upload identifiable, confidential, or unauthorised health information. For sensitive research data, use an approved secure environment and follow the relevant institutional rules.

## 13. What this project does not claim

I do not claim that this prototype:

- Diagnoses a disease or recommends treatment.
- Predicts clinical outcomes with validated accuracy.
- Processes raw scans or raw wearable sensor streams.
- Establishes a medically meaningful relationship from its synthetic scenarios.
- Demonstrates causal identification on real observational data.
- Proves fairness or robustness from the included descriptive analyses.
- Has been externally or clinically validated.

I keep these boundaries explicit so that the software can be inspected and improved without overstating what its current results demonstrate.
