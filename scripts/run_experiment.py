from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medprogress.experiment import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic multimodal disease-progression demo.")
    parser.add_argument("--patients", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario", choices=["no_signal", "progression_signal", "difficult"],
        default="progression_signal", help="Synthetic progression scenario to simulate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.patients < 30 or args.epochs < 1:
        parser.error("--patients must be >= 30 and --epochs must be >= 1")

    result = run_experiment(
        n_patients=args.patients, seed=args.seed, epochs=args.epochs, scenario=args.scenario
    )
    summary = {
        "synthetic_only": True,
        "settings": {
            "patients": args.patients, "epochs": args.epochs, "seed": args.seed,
            "scenario": args.scenario,
        },
        "synthetic_scenario": args.scenario,
        "software_versions": result["software_versions"],
        "split_sizes": {
            name: {
                "patients": int(frame.patient_id.nunique()),
                "forecast_windows": int(len(frame)),
            }
            for name, frame in [
                ("train", result["splits"].train),
                ("validation", result["splits"].validation),
                ("test", result["splits"].test),
            ]
        },
        "benchmarks": result["benchmarks"].to_dict(orient="records"),
        "neural_metrics": {
            "accuracy": result["neural_metrics"]["accuracy"],
            "macro_f1": result["neural_metrics"]["macro_f1"],
            "n_test_examples": result["neural_metrics"]["n_test_examples"],
        },
        "causal_simulation": result["causal"],
        "disclaimer": "Synthetic research demonstration only; not clinical evidence.",
    }
    text = json.dumps(summary, indent=2, default=float)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote summary to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
