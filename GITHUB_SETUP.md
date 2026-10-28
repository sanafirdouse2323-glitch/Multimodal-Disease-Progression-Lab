# Repository setup and maintenance

This project belongs in the existing repository:

https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab

## Clone and run

```bash
git clone https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab.git
cd Multimodal-Disease-Progression-Lab
python -m venv .venv
```

Activate the virtual environment, install dependencies, and start the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Run verification locally:

```bash
PYTHONPATH=src python -m pytest -q
python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --output /tmp/smoke-summary.json
```

## Commit future changes

After checking `git status` and running tests, commit and push only intended files:

```bash
git status --short
PYTHONPATH=src python -m pytest -q
git add README.md GITHUB_SETUP.md app.py src tests scripts requirements.txt .github .gitignore examples
git commit -m "Describe the verified change"
git push
```

Never commit real patient data, credentials, environment files, or model artifacts containing private data. This demonstration uses synthetic data only and is not a clinical tool.
