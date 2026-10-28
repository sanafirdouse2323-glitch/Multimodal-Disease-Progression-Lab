# Uploading Your Own Tabular Data

The Streamlit app offers three modes: **Synthetic demo**, **Upload CSV**, and **Empty CSV template**. The blank template contains column headers only. Fill it in, then choose **Upload CSV**.

## Important scope and privacy

The upload workflow accepts **tabular, numerical, feature-level data**. It does not load raw medical images, MRI or X-ray files, or raw wearable sensor streams, and it does not extract features from those files. Imaging-related columns should contain numerical features already extracted by a method you have permission to use.

The public Render demo receives the file you upload. **Do not upload names, dates of birth, medical record numbers, identifiable health records, confidential information, or data you are not authorised to use.** For sensitive or regulated research data, run the software locally inside your institution's approved secure environment and follow the relevant ethics, consent, and data-governance requirements.

Format validation only checks whether the table can be parsed and passed to this demo. It does not confirm that the data are clinically meaningful, legally usable, unbiased, or suitable for a research conclusion.

## One row per visit

Each row represents one visit for one participant. A participant must have at least two visits. Use the same non-identifying study code for the same participant across visits. The app needs at least 10 unique participants to create its train/validation/test split; cohorts under 30 participants are flagged as especially small. Larger, justified samples are normally needed for stable research estimates.

## Columns

### Required columns

| Column | Meaning and rule |
|---|---|
| `patient_id` | A non-identifying participant code, for example `SUBJECT_001`. Do not use a person's name or medical record number. |
| `time_months` | Numeric months since the participant's baseline visit; use non-negative numbers and a distinct time for each visit from the same participant. A common pattern is 0, 6, 12. |
| `current_stage` | The stage at this visit, encoded as an integer 0, 1, 2, or 3. If your project uses different labels, create and document a defensible mapping before upload. The demo predicts the label at the next observed visit. |

### Optional numerical columns

| Column(s) | Meaning and rule |
|---|---|
| `age_years` | Age in years, when appropriate and permitted; allowed values are 0–120. Blank is allowed. |
| `sex_binary` | Optional group code 0 or 1. Keep the meaning of each code consistent and documented in your study. The app uses these codes only to display descriptive subgroup metrics; it does not infer what each code means. |
| `lab_a`, `lab_b` | Two generic numerical measurement columns for the demo. Rename/map your permitted numerical measurements to these columns, and document what they mean in your research notes. |
| `img_0`–`img_7` | Eight numerical imaging-related features, already extracted from an image by an appropriate pipeline. These are not image pixels or files. |
| `wear_0`–`wear_5` | Six numerical wearable-related features, computed from an appropriate data-processing pipeline. These are not raw sensor streams. |

Optional columns may be omitted, and blank feature cells are permitted. If a complete imaging or wearable feature group is absent for a row, the app sets the corresponding missing-modality indicator. The app derives scaled age, scaled current stage, time gap, and missing-modality indicators from the uploaded values.

Extra columns are ignored, and the app displays a warning listing them. Rename only columns that genuinely correspond to the template's meaning; do not map unrelated variables simply to make the file pass validation.

## How uploaded data are used

- Progression models, modality comparisons, added missingness stress tests, and subgroup metrics use the uploaded visit table after validation.
- Training, validation, and test partitions are separated by participant ID, not by individual visits.
- The causal-effect tab intentionally uses a separate synthetic treatment-effect simulation. It does **not** estimate a treatment effect from the uploaded CSV.
- Results are exploratory model outputs. This demo does not validate a clinical prediction tool, produce a diagnosis, or recommend treatment.
