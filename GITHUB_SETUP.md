# How I Set Up and Maintain This Repository

I maintain this project in the existing GitHub repository:

https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab

I use feature branches and pull requests so that I can inspect changes and run the automated checks before they reach main. Render is connected to main, so merging a change can trigger a public deployment.

## 1. Clone the repository

    git clone https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab.git
    cd Multimodal-Disease-Progression-Lab

I use Python 3.12, matching the version recorded in .python-version.

## 2. Create and activate a virtual environment

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

macOS or Linux:

    python -m venv .venv
    source .venv/bin/activate

If PowerShell blocks activation because of the local execution policy, I can use the environment's Python directly instead:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run app.py

## 3. Install the project dependencies

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

The requirements file uses CPU-only PyTorch. I do not require a CUDA-capable GPU to run this demonstration.

## 4. Run the website locally

    streamlit run app.py

I open the local URL printed in the terminal. The default data source is the synthetic demo, so I can test the workflow without uploading a CSV.

## 5. Run the automated checks

    PYTHONPATH=src python -m pytest -q

Then I run a small end-to-end synthetic experiment:

    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario progression_signal --output smoke-summary.json

I can also check the other data-generator scenarios:

    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario no_signal
    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario difficult

For a more thorough model comparison, I run:

    python scripts/evaluate_progression_scenarios.py --patients 400 --seeds 10 --epochs 5 --bootstrap-replicates 500 --runs-output progression_scenario_runs.csv --summary-output progression_scenario_summary.csv

The repeated-seed evaluation produces two CSV files. They are generated results, not source code. I inspect them before making any performance statement and do not commit them unless I intend to preserve a particular experiment output.

## 6. Make a change safely

I create a separate branch before editing:

    git switch main
    git pull origin main
    git switch -c docs/describe-my-change

I replace the example branch name with a short name that describes the task. I then edit only the files required for that task.

Before committing, I inspect the exact changes and run the checks:

    git status --short
    git diff --check
    git diff
    PYTHONPATH=src python -m pytest -q

I stage only the files I intended to change. For example:

    git add README.md app.py

I then commit and push the feature branch:

    git commit -m "Describe the verified change"
    git push -u origin docs/describe-my-change

I open a pull request from that branch into main. I inspect the file list and diff, review the GitHub Actions results, and merge only when the change is ready. I do not use a broad git add command if it could accidentally include generated outputs or private files.

## 7. Understand the automated workflow

The GitHub Actions workflow in .github/workflows/tests.yml runs for pull requests targeting main and pushes to main. It:

1. Sets up Python 3.12.
2. Installs requirements.txt.
3. Runs the pytest suite.
4. Runs a small synthetic smoke experiment.
5. Runs the repeated-seed comparison across the three synthetic scenarios.

A green workflow means those configured checks completed successfully for that commit. It does not establish clinical validity or prove that a model outperforms the persistence baseline on real data.

## 8. Understand what happens after a merge

My Render service is connected to main and automatic deployment is enabled. A merge to main should trigger a deployment. I still verify the deployment in Render and open the public website to check the changes. I do not equate a merged pull request with a confirmed live deployment.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the service configuration and live-site verification checklist.

## 9. Protect data and credentials

I do not commit:

- Real patient records or identifiable health information.
- Private or confidential datasets.
- API keys, passwords, tokens, private keys, or environment files containing secrets.
- Generated experiment outputs unless I intentionally decide to preserve a non-sensitive result.

The public hosted app receives uploaded CSV files. I direct visitors not to upload identifiable, confidential, or unauthorised health data. Sensitive research data should only be processed in an approved secure environment.

## 10. Project documentation

- [README.md](README.md) is the main project overview and quick-start guide.
- [METHODS_AND_LIMITATIONS.md](METHODS_AND_LIMITATIONS.md) explains the task, models, evaluation design, synthetic scenarios, and limitations.
- [UPLOAD_DATA_GUIDE.md](UPLOAD_DATA_GUIDE.md) explains the CSV schema and upload process.
- [DEPLOYMENT.md](DEPLOYMENT.md) records the Render deployment workflow.
