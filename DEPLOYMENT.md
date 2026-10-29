# Deployment

## Live demo

**Public app:** https://multimodal-disease-progression-lab.onrender.com

**Render service:** https://dashboard.render.com/web/srv-dbgsmsfb9u1s73963g4g

The service was created from this repository's `main` branch. Render reported a successful build and deploy, and the service logs show Streamlit starting on the Render-assigned port.

The service uses Render's free plan in Singapore. Free services may sleep when idle, so the first visit after inactivity can take longer to respond. If the page does not load, check the Render service dashboard and recent deploy/log events.

## Current deployment configuration

- Runtime: Python 3.12 (from `.python-version`)
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Branch: `main`
- Dependencies: CPU-only PyTorch to avoid a CUDA download

Auto-deploy is enabled, so new commits to `main` initiate a Render deployment. Check the Render dashboard and deployment events before assuming a new revision is live.

## Verify locally

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --output /tmp/smoke-summary.json
streamlit run app.py
```

For an initial live check, open the public app, choose 120 synthetic patients and 5 neural epochs, click **Run full experiment**, and confirm the six tabs, benchmark table, missingness chart, causal estimates, subgroup metrics, and CSV/JSON downloads.

The built-in cohort and causal-effect tab use synthetic data. When a CSV is uploaded, progression modelling, modality comparisons, missing-modality stress tests, and subgroup metrics use the uploaded feature-level table; the causal-effect tab remains a separate synthetic simulation. This is an educational research demonstration, not a clinical tool. Never commit patient records, credentials, or identifiable health information.


## CSV upload and empty template

The app provides three modes in the sidebar: **Synthetic demo**, **Upload CSV**, and **Empty CSV template**. The blank template has headers only. Uploaded files must be a longitudinal table with one row per visit and required columns `patient_id`, `time_months`, and `current_stage` (integer codes 0–3). Optional numerical feature columns are documented in `UPLOAD_DATA_GUIDE.md`.

The public hosted app receives uploaded files. Never upload identifiable or confidential health data. The uploader accepts numerical feature-level columns only, not raw medical images or raw wearable sensor files. The validation checks format and minimum structural requirements; it does not approve a clinical dataset or prove clinical validity. The causal-effect tab always uses a separate synthetic simulation, even when a progression CSV is uploaded.
