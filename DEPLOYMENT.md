# How I Deploy and Verify the Website

I host the Streamlit dashboard on Render and connect the service to this GitHub repository. This file records the deployment workflow and the settings I have verified. I use the Render dashboard and deployment records to confirm what is actually live.

## Public website and service

- **Live application:** https://multimodal-disease-progression-lab.onrender.com
- **GitHub repository:** https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab
- **Render service dashboard:** https://dashboard.render.com/web/srv-dbgsmsfb9u1s73963g4g

## Deployment settings

The Render service is connected to the repository's **main** branch. The configuration I have verified is:

| Setting | Configured value |
|---|---|
| Runtime | Python |
| Python version | 3.12, from .python-version |
| Build command | pip install -r requirements.txt |
| Start command | streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true |
| Branch | main |
| Automatic deployment | Enabled for commits to main |
| Dependency note | The requirements file installs CPU-only PyTorch. |

The service is on Render's free plan in the Singapore region in the configuration I inspected. Free services may sleep when idle, so the first request after inactivity may take longer to respond.

I do not assume that a GitHub commit or merge means the new version is already live. I check the Render deployment history and confirm that the latest successful live deployment points to the intended main-branch commit.

## My deployment workflow

1. I make and test changes on a feature branch.
2. I open a pull request targeting main.
3. I review the changed files and wait for the GitHub Actions checks to finish.
4. I merge only after reviewing the code and test results.
5. Because automatic deployment is enabled, a commit to main should trigger a Render deployment.
6. I open the Render service dashboard and check the deployment status, deployed commit, and logs.
7. I open the public website and exercise the changed workflow before I describe it as live.

If the deployment does not start or fails, I inspect the Render event and build/runtime logs. I do not change the connected repository, branch, build command, or start command as a first troubleshooting step; I verify the existing configuration first.

## Local verification before I merge

From the repository root, I run:

    python -m pip install -r requirements.txt
Windows PowerShell:

    $env:PYTHONPATH = "src"
    python -m pytest -q

macOS or Linux:

    PYTHONPATH=src python -m pytest -q

Then I run the smoke experiment on either platform:

    python scripts/run_experiment.py --patients 120 --epochs 3 --seed 42 --scenario progression_signal --output smoke-summary.json
    streamlit run app.py

For the full synthetic scenario comparison, I run:

    python scripts/evaluate_progression_scenarios.py --patients 400 --seeds 10 --epochs 5 --bootstrap-replicates 500 --runs-output progression_scenario_runs.csv --summary-output progression_scenario_summary.csv

The full scenario evaluation takes longer than the ordinary smoke experiment because it repeats the comparisons over multiple seeds and scenarios. I use it to inspect model behaviour; a successful run is not proof that a model is clinically useful.

## What I check in the live app

After a successful deployment, I check that:

- The homepage guide and data-source options load.
- The synthetic demo can run using the default progression-signal scenario.
- All three synthetic scenarios can be selected and run.
- The six result tabs appear after a successful experiment.
- The benchmark table and changed-stage diagnostics render.
- The modality comparison and additional missing-modality stress test render.
- The causal simulation and descriptive subgroup table render.
- The data preview and CSV/JSON downloads work.
- The CSV template can be downloaded and the upload instructions are visible.

I use a non-sensitive test file when checking the upload flow. I never use real patient information for a public deployment smoke test.

## What the website does with uploaded data

The app accepts a longitudinal CSV containing numerical feature-level data. It does not analyse raw image files or raw wearable streams. The hosted service receives the uploaded file, so I tell visitors not to upload identifiable, confidential, or unauthorised health information.

When a CSV is uploaded, it is used for progression modelling, modality comparisons, the added missingness stress test, and subgroup summaries. The causal-effect tab remains a separate synthetic simulation. Passing the CSV validator does not establish clinical suitability.

This deployment is an educational research demonstration. It is not approved for diagnosis, prognosis, or treatment decisions.
