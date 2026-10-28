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

All data are synthetic. The app is an educational research demonstration, not a clinical tool. Never commit patient records, credentials, or identifiable health information.
