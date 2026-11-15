# Project Overview: Three Related Research Prototypes

**Purpose of this page:** explain how three separate repositories relate, which one to show first, and what each repository does and does not demonstrate.

These repositories are related to my exploration of multimodal machine learning and disease-progression methods for the University of Bern PhD in Machine Learning in Medicine. They are **three separate prototypes**, not one integrated software system. They do not share a runtime pipeline, and their synthetic datasets, prediction targets, experiments, and reported metrics are different.

## Recommended starting point

### 1. Main demonstration — Multimodal Disease Progression Lab

- **Repository:** [Multimodal-Disease-Progression-Lab](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab)
- **Interactive demonstration:** [Open the hosted Streamlit app](https://multimodal-disease-progression-lab.onrender.com)

This is the most complete, visitor-facing demonstration of the three. It contains a Streamlit dashboard and a Python experiment pipeline. Its default workflow uses synthetic longitudinal participant data and compares imaging-related numerical features, wearable-style numerical features, and clinical/visit-level features.

The principal prediction task is **next-observed-visit classification**: features from one visit are used to predict the generic stage code at that participant's next recorded visit. Visit intervals can vary. The stage codes 0–3 are generic labels, not a validated medical staging scale.

The repository also contains:
- Random Forest, Logistic Regression, neural multimodal fusion, persistence, and majority-class comparisons;
- participant-level train/validation/test splitting and training-only preprocessing for the neural model;
- an inference-time missing-modality masking exercise;
- descriptive subgroup summaries; and
- a separate synthetic treatment-effect exercise.

**Important boundaries:** the app does not process raw MRI/X-ray images or raw wearable sensor streams. Uploaded modality columns must contain pre-computed numerical features. The causal exercise uses a separate synthetic dataset; it does not estimate treatment effects from the progression cohort or an uploaded CSV. Subgroup summaries are descriptive, not a formal fairness audit.

The README records a repeated-seed synthetic evaluation in which persistence had higher mean overall accuracy than the multimodal Random Forest, Logistic Regression, and neural fusion model in all three reported scenarios. The neural model performed substantially worse in those runs. This negative result is part of the project's current evidence and should be discussed honestly; the prototype should not be presented as an effective or clinically validated prediction model.

### 2. Supporting implementation prototype — Multimodal Disease-Progression Prototype 2

- **Repository:** [multimodal-disease-progression-prototype-2](https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-prototype-2)

This is a smaller, standalone Python/PyTorch baseline focused on implementation of modality-specific feed-forward encoders and a fusion model. It generates its own synthetic cohort, constructs a next-observed-visit stage-classification task, includes an auxiliary current-stage prediction head, uses patient-level splitting and training-only preprocessing, and can save a checkpoint, metrics, and test predictions.

Its documented scope explicitly says that causal treatment-effect estimation and fairness mitigation are **not implemented** in this prototype. It is useful when discussing the PyTorch model implementation, but it is not the same application as the main Lab and is not a replacement for the hosted dashboard.

### 3. Supporting methods sandbox — Multimodal disease progression: a basic Python demo

- **Repository:** [multimodal-progression-basic](https://github.com/sanafirdouse2323-glitch/multimodal-progression-basic)

This is a compact, script-based set of five separate synthetic experiments:
1. modality-specific models and early/late fusion;
2. sparse longitudinal progression using simulated repeated measurements;
3. causal treatment-effect estimation under stated simulation assumptions;
4. missing-modality experiments, including modality-dropout training; and
5. subgroup performance comparisons and reweighting.

It is useful for explaining the range of methodological questions explored in the initial experiments. Its experiments use their own simulated data and, in places, different outcomes and evaluation designs from the main Lab. Its reported AUC, progression-rate, causal-bias, and subgroup values must **not** be compared directly with the Lab's next-visit classification metrics or treated as results from one shared benchmark.

The README itself describes the missing-modality and fairness effects as small and dependent on the chosen simulation, and states the causal experiment's no-unmeasured-confounding assumption.

## How to present the repositories

For a short PhD discussion, I recommend this order:

1. Open the **Multimodal Disease Progression Lab** and explain the workflow, the next-observed-visit target, the baselines, and the limitations.
2. Show **Prototype 2** only when the discussion turns to the PyTorch implementation and separate modality encoders.
3. Show the **basic Python demo** when asked how the initial methodological ideas—causal estimation, sparse longitudinal data, missing modalities, and subgroup comparisons—were explored.

A concise and accurate description is:

> I have developed three related but separate research prototypes exploring multimodal machine learning and disease-progression methods using synthetic data. The Multimodal Disease Progression Lab is my main interactive demonstration. A separate PyTorch prototype focuses on modality-specific encoders and neural fusion, while a smaller Python sandbox contains exploratory simulations of multimodal fusion, sparse longitudinal data, causal estimation, missing modalities, and subgroup comparisons. The repositories use different synthetic setups and are not one integrated pipeline. I treat them as methodological prototypes rather than clinically validated systems, and I report the current baseline results and limitations transparently.

## Shared scientific and safety boundaries

- All three repositories use synthetic data for the experiments described in their READMEs; their scores are not clinical evidence.
- Generic simulated labels and latent variables do not establish real disease biology or a clinically meaningful progression scale.
- None of these repositories establishes clinical utility, external validity, or readiness for patient care.
- The main Lab's public hosted app accepts uploaded files. Do not upload identifiable, confidential, or unauthorised health information.
- Results from different repositories should not be pooled or ranked unless a common dataset, target definition, split protocol, baselines, and evaluation procedure are deliberately established.

## Repository links

- [Main interactive Lab repository](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab)
- [Live Lab demonstration](https://multimodal-disease-progression-lab.onrender.com)
- [PyTorch implementation prototype](https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-prototype-2)
- [Basic methods sandbox](https://github.com/sanafirdouse2323-glitch/multimodal-progression-basic)

---

**Scope of this overview:** this page describes the repositories as inspected on 15 November 2026. It is a navigation and positioning guide, not an independent scientific validation, clinical evaluation, or claim that the prototypes reproduce the University of Bern lab's research.
