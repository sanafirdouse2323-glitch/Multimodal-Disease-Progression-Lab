# Deploy the dashboard

The app entry point is `app.py` on the `main` branch. The repository is public and includes `requirements.txt` and a Python version file.

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that owns this repository. Authorize Streamlit to access public repositories if prompted.
2. Choose **Create app**.
3. Select repository `sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab`, branch `main`, and main file path `app.py`.
4. Choose a memorable app subdomain if available, then select **Deploy**.
5. Wait for the initial build to finish and open the actual URL shown by Streamlit. Do not guess a `.streamlit.app` URL before deployment creates one.
6. Test the running app: choose 120 synthetic patients and 5 epochs, click **Run full experiment**, and verify that each tab renders and the CSV/JSON download controls work.

Deployment requires the repository owner's authenticated action in Streamlit Community Cloud. This repository change alone does not create a live hosted app.

## Local smoke test

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
streamlit run app.py
```

The `requirements.txt` uses CPU-only PyTorch because this demo has no CUDA/GPU requirement. The dashboard uses synthetic data only, with no patient records or external credentials. It is an educational research demonstration, not a clinical tool.
