# Using My CSV Upload Option

I designed the homepage so visitors can start in three ways: try sample data, enter simple whole numbers, or upload a spreadsheet. This guide explains the spreadsheet formats. Visitors who do not already have a research CSV can choose **Enter simple numbers** instead.

Please read the privacy and scope notes before uploading anything.

## The simple number sheet

I provide a beginner-friendly four-column format for people who want to work with numbers rather than a detailed research CSV.

| Column | What to enter |
|---|---|
| **Person number** | A different positive whole number for each person, such as 1, 2, 3. |
| **Month 0 stage** | The stage label recorded at the starting visit. |
| **Month 6 stage** | The stage label recorded at month 6. |
| **Month 12 stage** | The stage label recorded at month 12. |

Use whole numbers only—no decimal points. Stage labels must be 0, 1, 2, or 3. These are generic category labels for this demonstration; they are not diagnoses or a validated medical scale. Each row is one person, with three visits at months 0, 6, and 12.

The homepage starts this table with 30 artificial example rows. You can edit the numbers, add or remove rows, or download the blank number sheet. You can upload that same four-column sheet through **Upload a spreadsheet**; the app recognises it and converts it into visit records.

**Important scope note:** the simple number sheet contains visit times and stage labels only. It does not contain image or wearable-device measurements. Results from this route are not a full multimodal analysis. To compare those data groups, use the detailed CSV format below and supply pre-computed numerical features.

## What I accept—and what I do not accept

I accept a **CSV table with one row per participant visit**. The table can contain stage labels, visit times, and optional numerical features.

I do not accept or process raw medical images, MRI files, X-ray files, image folders, raw wearable sensor recordings, or other raw media. The columns named img_0 to img_7 and wear_0 to wear_5 must contain numerical features that have already been extracted using a method you are authorised to use. This app does not perform image feature extraction or raw sensor processing.

The uploader currently has a 20 MB file-size limit.

## Privacy and research responsibility

The public Render website receives the file uploaded through the browser. **Please do not upload names, dates of birth, medical record numbers, identifiable health records, confidential information, or data that you are not authorised to use.**

For sensitive or regulated research data, I recommend running the repository locally inside an institutionally approved secure environment and following the relevant ethics, consent, security, and data-governance requirements.

The validation code checks the file structure and value formats. It does not confirm that the dataset is clinically meaningful, representative, unbiased, legally usable, or appropriate for a research conclusion. Passing validation is not the same as ethical approval or clinical validation.

## How I expect the CSV to be organised

Each row represents one visit for one participant. Use the same non-identifying study code for every visit belonging to the same participant. A participant must have at least two visits so that the app can create a current-visit/next-visit example.

The application requires at least 10 unique participants to create the train, validation, and test partitions. This is a technical minimum, not a recommended research sample size. I flag cohorts below 30 participants as small, and a larger, justified sample is normally needed for stable research estimates.

The app sorts each participant's visits by time. Each participant must have distinct visit times. Use non-negative numeric months from that participant's baseline visit, such as 0, 6, and 12.

The CSV reader initially reads values as text so that participant codes such as 001 are not silently converted to the number 1. I then validate and convert the documented numeric columns.

## Required columns

| Column | What I expect |
|---|---|
| **patient_id** | A non-identifying participant code, such as SUBJECT_001. Do not use a person's name or medical record number. Keep the same code across that participant's visits. |
| **time_months** | Numeric months since that participant's baseline visit. Values must be zero or positive. A participant cannot have two visits at the same time. |
| **current_stage** | An integer code of 0, 1, 2, or 3 for the stage recorded at that visit. If your project uses a different staging system, define and document a defensible mapping before uploading. |

The model target is the stage at the **next observed visit**. It is not a fixed 6-month or 12-month target. The software does not know what the stage codes mean medically.

## Optional numerical columns

| Column | What I use it for |
|---|---|
| **age_years** | Optional age in years. Nonblank values must be between 0 and 120. |
| **sex_binary** | Optional numeric group code. Nonblank values must be 0 or 1. I use these values only to form descriptive groups; the app does not infer what either code means. Keep the mapping consistent and document it outside the app. |
| **lab_a, lab_b** | Two generic numerical measurement fields. Map measurements to these names only when their meaning is genuinely appropriate, and keep a record of the mapping. |
| **img_0 to img_7** | Eight numerical imaging-related features that have already been extracted from images. These columns are not image pixels or image files. |
| **wear_0 to wear_5** | Six numerical wearable-related features that have already been computed from sensor data. These columns are not raw sensor streams. |

You may omit optional columns or leave feature values blank. If all values for an imaging or wearable group are missing in a row, the app marks that modality as missing. During model fitting, the implemented preprocessing pipelines handle missing numerical feature values using training-data imputation.

Extra columns are ignored and listed in a warning. I do not recommend renaming unrelated variables merely to make a file pass validation.

## How I validate the file

The current validator checks that:

- The file can be read as a CSV and has unique column names.
- The required columns are present.
- Participant IDs are not blank.
- Time, stage, and optional feature columns contain numeric values or permitted blanks.
- Time values are non-negative.
- Stage values are integer codes 0, 1, 2, or 3.
- Age values, when supplied, are between 0 and 120.
- The optional group code uses 0 or 1.
- At least 10 unique participants are present.
- Every participant has at least two visits.
- A participant does not have duplicate visit times.

A file can pass these structural checks and still fail when I start the experiment. The training partition must contain at least two different next-visit stage classes for the classifiers to train. A small sample, rare stage labels, or limited variation can make estimates unstable.

## What happens after upload

1. I read the CSV and validate its structure.
2. I convert the accepted columns into the feature format used by the experiment.
3. I create next-observed-visit examples from consecutive visits for each participant.
4. I split participants—not individual visit rows—into training, validation, and test partitions.
5. I fit and evaluate the progression models on the uploaded feature-level cohort.
6. I show the model comparisons, modality results, added missingness stress test, and descriptive subgroup metrics.

The causal-effect tab is different: it always uses a separate synthetic treatment-effect simulation. **It does not estimate a treatment effect from the CSV you upload.**

## How to use the detailed CSV option

1. Choose **Upload CSV** in the centre of the homepage.
2. Select **Download empty CSV template** in the centre panel to get the supported column names.
3. Prepare one row per participant visit, using the detailed required and optional columns above.
4. Choose your completed CSV file and read the validation message and preview.
5. Select **Run on my spreadsheet**.
6. Review the results and their limitations before interpreting them.

The detailed template contains column names only. It is intended for users who already have a visit-level feature table. Visitors who only have simple stage numbers can use the separate four-column number sheet instead.

## What I do with the uploaded file

The current app uses the uploaded cohort for next-visit progression modelling, modality comparisons, missing-modality stress tests, and subgroup summaries. It does not use the uploaded cohort for the separate causal-effect simulation.

I do not claim that format checks establish clinical validity, fairness, causality, or external generalisability. I treat the output as exploratory software results, not a diagnosis, prognosis for patient care, or treatment recommendation.
