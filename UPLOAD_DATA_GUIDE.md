# CSV Upload Guide

This page explains the two CSV formats accepted by the dashboard. For the prediction task, model inputs, metrics, and result tabs, see [DATA_AND_RESULTS_GUIDE.md](DATA_AND_RESULTS_GUIDE.md).

Before uploading, remember that the hosted website is public. Do not upload names, dates of birth, medical record numbers, identifiable or confidential health information, or data you are not authorised to use.

## Choose a format

The app accepts CSV files up to 20 MB. It supports two formats.

### 1. Simple number sheet

Use one row per person and these exact column headings:

| Column | Meaning |
|---|---|
| `Person number` | A unique positive whole number for each person. |
| `Month 0 stage` | Stage code at month 0. |
| `Month 6 stage` | Stage code at month 6. |
| `Month 12 stage` | Stage code at month 12. |

Enter whole numbers only. Stage codes must be 0, 1, 2, or 3. Each person number must be unique, and every cell must be filled. At least 10 people are required.

The homepage opens with 30 pre-filled artificial examples. You can edit that table or download the blank four-column sheet. The app also recognises a CSV containing these four headings and converts each row into visits at months 0, 6, and 12.

This format contains only visit times and stage codes. It does not provide imaging, wearable, age, or laboratory features, so it is not a multimodal input.

### 2. Detailed visit-level CSV

Use one row per participant visit. Keep the same non-identifying `patient_id` for all visits belonging to one person.

**Required columns**

| Column | Requirement |
|---|---|
| `patient_id` | Nonblank participant code. Do not use names or medical record numbers. |
| `time_months` | Numeric months since that participant's baseline visit; must be zero or positive. |
| `current_stage` | Integer stage code: 0, 1, 2, or 3. |

**Optional numerical columns**

| Column | Accepted values |
|---|---|
| `age_years` | Blank or a number from 0 to 120. |
| `sex_binary` | Blank, 0, or 1. Document what the codes mean in your own data records; the app does not infer their meaning. |
| `lab_a`, `lab_b` | Blank or numeric values. Units and meanings are not defined by the app. |
| `img_0` to `img_7` | Blank or pre-computed numerical imaging-related features. |
| `wear_0` to `wear_5` | Blank or pre-computed numerical wearable-related features. |

The image and wearable columns are feature values, not image files, pixel data, or raw sensor recordings. The app does not extract features from raw medical images or process raw wearable streams. Optional columns may be omitted; extra columns are ignored and reported in an upload warning.

## Validation rules

For a detailed CSV, the app checks that:

- The required columns are present and column names are unique.
- Participant IDs are not blank.
- Numeric fields contain numbers or permitted blanks; infinite values are rejected.
- Visit times are non-negative.
- Stage codes are whole numbers from 0 to 3.
- Nonblank ages are between 0 and 120, and nonblank `sex_binary` values are 0 or 1.
- There are at least 10 unique participants.
- Every participant has at least two visits.
- A participant has no duplicate visit times.

The CSV reader initially treats values as text so identifiers such as `001` are not silently converted to `1`. Numeric fields are converted after the headers and values are checked.

The app then creates consecutive-visit prediction examples and makes a seeded, participant-level 60/20/20 train/validation/test split. The split is not stratified by stage. The training partition must contain at least two different next-visit stage codes for the classifiers to run; the app does not guarantee that every stage appears in validation or test. The minimum of 10 participants is a software requirement, not a scientifically adequate sample-size recommendation.

Passing file validation confirms only that the input meets these structural checks. It does not establish that the data are clinically meaningful, representative, unbiased, ethically approved, or legally usable.

## Upload steps

1. Select **Upload CSV** on the homepage.
2. Use **Download empty CSV template** for the detailed visit-level format, or prepare the four-column simple number sheet.
3. Select your CSV file and review the validation message and preview.
4. Select **Run on my spreadsheet**.
5. Review the output alongside its limitations.

For either format, progression analysis uses the supplied visit and feature values. The causal-effect tab always uses a separate synthetic dataset; it does not estimate a treatment effect from your uploaded CSV.

## Privacy and responsible use

The public Render website receives the uploaded file. Use only non-identifying participant codes and upload data only when you have the necessary permissions. For sensitive or regulated research data, use an institutionally approved secure environment and follow the relevant ethics, consent, security, and data-governance requirements.

Downloaded prediction files include participant codes from the uploaded data. Handle exported files under the same data-protection rules as the source dataset.
