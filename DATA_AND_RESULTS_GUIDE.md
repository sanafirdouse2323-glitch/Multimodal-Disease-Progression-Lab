# Understanding the Data, Models, and Results

This guide explains what goes into the Multimodal Disease Progression Lab, how the software prepares the data, and what each result means. It describes the current implementation as it exists in this repository. It does not describe planned features as though they were already available.

> **Important:** This is an educational research prototype, not a diagnostic, prognostic, or treatment tool. Stage codes 0–3 are generic labels in this software. They are not a recognised medical scale. The synthetic records are artificial and do not represent real patients or established disease mechanisms.

## 1. Choose a data source

The homepage offers three ways to provide data.

### Synthetic demo

The application creates an artificial longitudinal cohort. The default sample size is 400 people. The sample-size setting counts people, not visit rows or prediction examples.

Each generated person has 3 to 6 visits. The time between consecutive visits is selected from 3, 6, 9, or 12 months. Each visit contains a generic stage code and generated numerical fields representing imaging-related, wearable-style, and clinical measurements. Some feature values are deliberately missing.

The default scenario is **Example with a pattern**. It is useful for inspecting the workflow, but the pattern is an assumption built into the generator. It is not a finding about a real disease.

The preview shown on the homepage uses 30 generated people and seed 42 to provide a stable example of the data structure. It illustrates the selected scenario's columns and one participant's next-visit rows. The preview is an example only; changing the sample size, random seed, or scenario changes the experiment data.

### Enter simple numbers

This option begins with an editable four-column table. Each row represents one person:

| Column | Meaning |
|---|---|
| Person number | A positive, unique number used to keep that person's three visits together. |
| Month 0 stage | The generic stage code at month 0. |
| Month 6 stage | The generic stage code at month 6. |
| Month 12 stage | The generic stage code at month 12. |

Use whole numbers. Stage codes must be 0, 1, 2, or 3. The pre-filled rows are artificial examples; visitors may edit them. The software converts the three columns of stage codes into three visit records per person.

This route contains visit times and stage codes only. It does not supply imaging measurements, wearable measurements, age, or laboratory measurements. Results from this route must not be described as a full multimodal analysis.

### Upload CSV

The detailed upload option accepts a CSV with one row per participant visit. The required columns are:

| Column | Required? | Meaning and checks |
|---|---|---|
| patient_id | Yes | A non-identifying study code. Use the same code for all visits belonging to one participant. It cannot be blank. |
| time_months | Yes | Numeric months since that participant's baseline visit. Must be zero or positive. A participant cannot have two rows with the same visit time. |
| current_stage | Yes | An integer code of 0, 1, 2, or 3 recorded at that visit. The software does not define the medical meaning of these codes. |
| age_years | No | Age in years. Nonblank values must be between 0 and 120. |
| sex_binary | No | A numeric group code of 0 or 1. The software does not infer what either code represents. Keep its meaning and mapping documented separately. |
| lab_a, lab_b | No | Two generic numerical measurement fields. Their units and clinical meanings are not defined by this prototype. |
| img_0 to img_7 | No | Eight pre-computed numerical imaging-related features. These are not image files or pixel values. |
| wear_0 to wear_5 | No | Six pre-computed numerical wearable-related features. These are not raw sensor streams. |

Optional columns may be omitted, and optional numerical values may be blank. Extra columns are ignored and identified in an upload warning. The file-size limit in the current uploader is 20 MB.

At least 10 unique participants are required by the current data-splitting code. Each participant must have at least two visits, and the training partition must contain at least two different next-visit stage classes for the classifiers to run. These are software requirements, not recommendations for a scientifically adequate sample size.

The CSV reader initially treats field values as text so identifiers such as 001 remain distinct from 1. Numeric fields are converted only after the headers and values have been checked.

## 2. What the synthetic data represent

The synthetic generator creates artificial measurements for software experiments. It does not simulate a named disease, and none of its generated values should be interpreted as real clinical measurements.

- **People and visits:** Each generated participant receives an artificial identifier such as SYNTH_000000 and between 3 and 6 visits.
- **Visit times:** The first visit is at month 0. Later gaps are randomly selected from 3, 6, 9, and 12 months. Participants can therefore have different follow-up schedules.
- **Stage codes:** The initial code is generated from an artificial participant-level risk value. Later codes may stay the same, move up by one, or move down by one under the scenario's transition rules. The code is always between 0 and 3. This is a generator rule, not a medical staging model.
- **Imaging-related features:** Eight numerical values, img_0 through img_7, are generated at each visit. Their artificial values depend on the generated stage, simulated participant characteristics, a small participant-specific pattern, scenario settings, and random noise. They are not generated images.
- **Wearable-style features:** Six numerical values, wear_0 through wear_5, are generated at each visit from an artificial relationship with stage, simulated participant characteristics, scenario settings, and random noise. They are not measurements from a real wearable device.
- **Clinical-style fields:** lab_a and lab_b are generic numerical values generated from artificial stage and participant-level variables with noise. They have no assigned units or clinical interpretation. Age is generated as an artificial value between 30 and 88 years. The numeric sex_binary code is generated as 0 or 1; it is not a realistic or validated demographic simulation.
- **Missing values:** In the standard scenarios, an entire imaging feature group is absent for a visit with probability 0.12, and an entire wearable feature group is absent with probability 0.18. Each of lab_a and lab_b is also independently left blank on some visits by the generator (7% per field). The difficult scenario raises the minimum imaging and wearable group missingness probabilities to 0.30 and 0.35.

### The three selectable scenarios

| Scenario shown in the dashboard | What the generator changes | How to interpret it |
|---|---|---|
| Example with a pattern (progression_signal) | A hidden artificial participant-level tendency influences upward stage-transition probability and is encoded noisily in some generated measurements. | A controlled experiment for checking whether a planted relationship can be learned. It is not a medical hypothesis. |
| Example without an extra pattern (no_signal) | Removes the additional link between the hidden participant-level progression tendency and the observed measurements. Stage-dependent random transitions and measurements related to current stage can still remain. | A negative control for that extra planted relationship. It does not mean every feature is unrelated to every stage label. |
| More difficult example (difficult) | Retains a weaker planted relationship, increases measurement noise, and increases imaging and wearable missingness. | A deliberately harder artificial setting. It is not an estimate of real-world missingness. |

## 3. How a row becomes a prediction example

The current task is **next-observed-visit classification**.

For each participant, the software sorts visits by time and pairs each visit with that participant's following recorded visit. The values at the earlier visit are the model inputs. The stage code at the following visit is the target the model tries to predict.

For example:

| Person | Current visit month | Months to next visit | Current stage | Next-visit target |
|---|---:|---:|---:|---:|
| SYNTH_000000 | 0 | 6 | 1 | 1 |
| SYNTH_000000 | 6 | 3 | 1 | 2 |

The values above illustrate the layout only; the live homepage preview displays rows generated by the code. In the first row, the model receives the month-0 information and is asked to predict the stage code recorded at the next visit. In the second row, it receives the month-6 information and predicts the code at the next recorded visit.

The intervals can differ. The software therefore does not predict a fixed 6-month or 12-month outcome. It predicts the code at the next observed visit. The target field is called target_stage in the internal prediction table.

## 4. Which fields are given to the models

The detailed CSV is converted into a common set of 22 numerical model features. The model uses the current visit's information, not the participant identifier or future visit's measurements.

| Model feature | Number of fields | How the software obtains or uses them |
|---|---:|---|
| img_0 to img_7 | 8 | Supplied pre-computed imaging-related values or generated synthetic values. |
| wear_0 to wear_5 | 6 | Supplied pre-computed wearable-related values or generated synthetic values. |
| age_scaled | 1 | age_years divided by 100. In synthetic data, age is generated by the artificial cohort generator. |
| sex_binary | 1 | The supplied or generated numeric 0/1 code. Its meaning is not inferred by the application. |
| lab_a, lab_b | 2 | Generic numeric fields with no units assigned by this software. |
| current_stage_scaled | 1 | current_stage divided by 3. |
| visit_gap_scaled | 1 | Months since the participant's previous visit divided by 12. It is 0 for the first visit. |
| image_missing | 1 | 1 when all eight imaging-related values are missing at that visit; otherwise 0. |
| wearable_missing | 1 | 1 when all six wearable-related values are missing at that visit; otherwise 0. |

The 22 fields above are the model inputs. Several other columns have administrative or target roles:

- **patient_id** groups visits, forms prediction examples, and keeps a participant in only one data partition. It is not a model input.
- **time_months** orders visits and is used to calculate time gaps. Absolute month is not directly included in the current model feature list.
- **next_interval_months** describes the interval to the next visit in the prediction-example table. It is not a current-visit model input.
- **target_stage** is the label from the following visit. It is the outcome to be predicted, not an input feature.
- **visit_idx** records visit order in the prepared cohort. It is not included in the model feature list.

The eight imaging-related and six wearable-related columns are generic feature slots. Their names do not tell us what anatomical structure, sensor signal, or clinical quantity they represent. A real research dataset requires a documented feature-extraction method, units, preprocessing, and provenance before these columns can be interpreted scientifically.

## 5. How participants are divided for evaluation

The software creates prediction examples first and then splits unique participant IDs into three groups:

| Partition | Approximate share of participants | Purpose |
|---|---:|---|
| Training | 60% | Fits the model parameters and training-data preprocessing values. |
| Validation | 20% | Selects the neural model checkpoint using macro-F1. |
| Test | 20% | Provides the held-out results displayed after the experiment. |

All visits and prediction examples from one participant remain in the same partition. This reduces leakage that could occur if visits from the same person appeared in both training and test data.

For the neural model, missing-value medians and standardisation values are calculated on the training partition and then applied to validation and test data. The test labels are not used to train the model or select its neural checkpoint.

A participant-level split is an important control, but it does not establish external validity, remove all bias, or make the cohort representative of a patient population.

## 6. Models and reference comparisons

The current application includes:

- **Persistence baseline:** predicts that the next stage code will be the same as the current stage code. It is a useful reference because stage labels may often remain unchanged.
- **Majority-class baseline:** always predicts the most frequent next-visit stage code in the training partition.
- **Random Forest:** trains models using each feature group separately and all feature groups together. The implementation uses 180 trees and median imputation.
- **Logistic Regression:** uses the same feature-group comparisons, with median imputation and standardisation.
- **Neural multimodal fusion:** processes the imaging-related, wearable-style, and clinical/missingness feature groups through separate small neural encoders. It combines those representations for four-class prediction. The default training setting is 20 epochs; the saved checkpoint is selected using validation macro-F1.

The application compares these models on the same held-out test examples. A more complicated model is not automatically better. The persistence comparison is particularly important when unchanged stage codes are common.

## 7. What the summary cards mean

After a successful run, the dashboard displays four summary cards:

| Card | Meaning |
|---|---|
| People in this example | Number of unique participants in the full selected cohort. |
| Visit-to-visit examples | Number of current-visit/next-visit pairs created from the cohort. A participant with more visits contributes more than one pair. |
| Examples checked | Number of prediction pairs in the held-out test partition. |
| Neural model score | The neural fusion model's test-set macro-F1, shown as a percentage. It is not the accuracy of every model and is not a measure of medical certainty. |

## 8. What the prediction scores mean

The prediction-results tab compares the fused model results with reference baselines. The main metrics are:

### Accuracy

Accuracy is the proportion of held-out prediction examples for which the predicted stage code exactly matches the next-visit target.

For example, 80% accuracy means that 80 out of every 100 test examples were assigned the correct generic stage code in that particular run. It does not mean the model is 80% medically accurate or that a person has an 80% chance of a particular outcome.

Accuracy can be high when most next-visit codes remain unchanged. This is why the persistence baseline and the changed-stage analysis are also shown.

### Macro-F1 (the balanced score)

For each of the four stage codes, the software calculates an F1 score from precision and recall, then takes the unweighted average across codes 0, 1, 2, and 3. This gives each code equal weight in the average, rather than allowing the most common code to dominate the score.

A higher macro-F1 indicates a better balance of precision and recall across the four codes in this test set. It is not a probability, a confidence level, or clinical validation. If a class is absent from the test data, the current implementation assigns that class an F1 contribution of zero.

### Training history

For the neural model, the training-history chart shows:

- **Training loss:** the model's average training error measure during each epoch. A downward trend can indicate that the model is fitting the training data; by itself, it does not show that performance on new participants is improving.
- **Validation macro-F1:** the balanced validation score after each epoch. The saved neural checkpoint is the epoch with the best validation macro-F1.

The test set remains separate from this checkpoint-selection process.

## 9. What the changed-stage table means

The changed-stage table asks whether a model notices that the stage code at the next visit differs from the current stage.

- **Visits where stage changed:** number of held-out prediction examples where target_stage differs from current_stage.
- **Exact next-stage accuracy:** among only those examples where the stage did change, the fraction for which the model predicted the exact next stage code.
- **Change detection precision:** among examples where the model predicted any change, the fraction where a change actually occurred.
- **Change detection recall:** among examples where a change actually occurred, the fraction the model identified as changed.
- **Change detection score (F1):** the harmonic mean of change-detection precision and recall.

Here, a "change" means any different code. The table does not distinguish upward from downward changes and does not identify a medically meaningful progression event. If no test example changed stage, exact accuracy on changed examples is undefined.

## 10. What the data-group comparison means

The Data groups tab compares Random Forest and Logistic Regression results using:

- Imaging-related features only.
- Wearable-style features only.
- Clinical and visit-related features.
- All feature groups together (multimodal fusion).

The chart uses macro-F1. It shows how these feature selections performed in the selected experiment. It does not establish that one data modality is generally more useful, more reliable, or more clinically important. Results can depend on sample size, missingness, feature definitions, scenario settings, and random seed.

The simple-number option does not contain imaging or wearable measurements. The dashboard therefore explains that a full comparison of those modalities is unavailable for that route.

## 11. What the missing-information experiment means

The Missing information tab uses a Random Forest trained on the original training partition. It then hides additional imaging-related or wearable-related values in the test partition and evaluates the same fitted model.

The setting shown on the horizontal axis is the additional masking probability. For each test row, the software independently draws whether to mask the imaging group and whether to mask the wearable group. At 0.50, each group has a 50% additional masking chance per test row; a row can have neither, one, or both groups masked. Existing missing values in the input are not made into observed values.

The model is not retrained for each masking level. The chart reports macro-F1 under this inference-time stress test. It is not an evaluation of all real-world missing-data mechanisms and does not prove robustness to missing-not-at-random data.

## 12. What the practice simulation means

The Practice simulation tab is a separate teaching example. It creates a new artificial treatment dataset (5,000 simulated individuals by default) with a known treatment effect of -0.8 outcome units. It does not use the progression CSV or the simple numbers entered on the homepage.

The simulated treatment assignment depends on artificial risk and age values, so a simple comparison of treated and untreated outcomes can be confounded. The tab compares:

- **Known simulated effect:** the treatment effect deliberately set by the generator (-0.8 outcome units).
- **Unadjusted treated-minus-control:** difference between the average simulated outcome of treated and untreated individuals.
- **Adjusted linear regression:** estimate from a regression that includes simulated treatment, risk, and age.
- **Inverse-probability weighting:** estimate using a fitted model for treatment assignment and weights based on its estimated propensity scores.
- **Estimated propensity-score range:** the lowest and highest estimated treatment probabilities after the implementation's clipping step.

This exercise checks estimators under a known artificial data-generating process. It does not estimate a treatment effect from uploaded patient data, and it does not prove that the assumptions needed for causal analysis hold in a real healthcare dataset.

## 13. What the group summaries mean

The Group summaries tab reports accuracy and macro-F1 for the available numeric code groups and age groups:

- Numeric code groups 0 and 1 are displayed as code groups. The software does not infer their real-world meaning.
- Age groups are split at the median age calculated from the training partition. The same threshold is applied to the test examples.
- The table shows the number of test examples in each group.

These results are descriptive comparisons. They are not a formal fairness audit or proof that a model treats demographic groups equitably. A meaningful fairness study needs documented group definitions, sufficient sample sizes, uncertainty estimates, clinically appropriate metrics, and careful consideration of data quality and intersectional groups.

## 14. What appears in My data and downloads

The final tab contains the visit data, prediction-example table, split sizes, and downloadable files.

- **Your visit information:** rows from the prepared cohort. For simple-number input, the displayed columns are person identifier, visit month, and stage code.
- **Visit-to-visit examples:** current visit, next-visit interval, current stage, and next-visit target. This makes the prediction task visible.
- **Split sizes:** counts of unique participants and prediction examples in training, validation, and test partitions.
- **Benchmark metrics (CSV):** model names, feature groups, accuracy, macro-F1, and number of test examples.
- **Held-out predictions (CSV):** test participant code, current visit month, next-visit interval, current stage, true next-stage target, and the predictions from the Random Forest, Logistic Regression, and neural fusion model.
- **Experiment summary (JSON):** run settings, software versions, split sizes, benchmark metrics, neural metrics, the separate synthetic causal-simulation summary, data mode, and a scope disclaimer.

Keep in mind that the uploaded CSV's identifiers are included in the exported held-out prediction file. Use only non-identifying study codes, and handle downloaded files according to your institution's data-protection requirements.

## 15. What the application does not currently provide

To keep the results clear, the following are not outputs of the current dashboard:

- It does not process raw MRI, X-ray, retinal, or other medical image files.
- It does not process raw wearable sensor time series.
- It does not predict a fixed-horizon outcome; it predicts the next observed stage code.
- It does not display calibrated individual prediction probabilities or personal prediction intervals.
- It does not estimate a causal treatment effect from the uploaded progression data.
- It does not provide a formal fairness certification or prove robustness to real-world missingness.
- It has not been clinically or externally validated and is not approved for patient care.

## 16. Privacy and responsible use

The hosted dashboard is public and receives any file uploaded through it. Do not upload names, dates of birth, medical record numbers, identifiable health records, confidential information, or data you are not authorised to use.

For sensitive or regulated research data, use an institutionally approved secure environment and follow the applicable ethics, consent, security, and data-governance requirements. A file passing the application's structural checks does not establish that it is scientifically suitable, legally usable, representative, or clinically valid.

## 17. A short glossary

| Term | Plain-language meaning in this application |
|---|---|
| Participant | One person represented by a participant code. |
| Visit | One recorded time point for a participant. |
| Prediction example | A pair consisting of one current visit and that participant's next recorded visit. |
| Feature | A numerical value supplied to a model as input. |
| Modality | A type of information, such as imaging-related or wearable-related measurements. |
| Target | The next-visit stage code the model is asked to predict. |
| Persistence | A reference prediction that repeats the current stage code. |
| Imputation | Replacing a missing numerical feature with a value calculated from the training data. |
| Macro-F1 | The unweighted average of the F1 score for the four stage codes. |
| Test partition | Participants held out for final evaluation after training and validation. |
| Synthetic data | Artificial records created by code rather than collected from real participants. |

For the CSV column rules, see [UPLOAD_DATA_GUIDE.md](UPLOAD_DATA_GUIDE.md). For implementation choices and limitations, see [METHODS_AND_LIMITATIONS.md](METHODS_AND_LIMITATIONS.md). For the comparison with the University of Bern PhD advertisement, see [RESEARCH_ALIGNMENT.md](RESEARCH_ALIGNMENT.md).
