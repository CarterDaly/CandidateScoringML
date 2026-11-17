"""Train the candidate-scoring model, log it to MLflow, and optionally register it with the API.

Labels follow a deliberately strict rule (the "hired" bar):
    followers >= 500 AND stars_received >= 300 AND public_repos >= 10 AND recency_days <= 730

Data sources:
    --synthetic N        generate N synthetic GitHub-like profiles (default, no network needed)
    --usernames FILE     fetch real public profiles for the usernames in FILE (one per line)
                         through the same GitHub connector the API uses; set GITHUB_TOKEN to
                         avoid the 60 requests/hour anonymous rate limit

Examples:
    python scripts/train_model.py
    python scripts/train_model.py --register http://localhost:5000
    python scripts/train_model.py --usernames my_accounts.txt --register http://localhost:5000

Outputs (git-ignored): training_data.csv, model.pkl, model_run_id.txt
"""
import argparse
import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEATURES = [
    "account_age_days",
    "followers",
    "public_repos",
    "stars_received",
    "language_diversity",
    "recency_days",
    "avg_repo_stars",
    "forked_ratio",
    "has_description_ratio",
    "watchers_received",
]


def hired_label(row) -> int:
    return int(
        row["followers"] >= 500
        and row["stars_received"] >= 300
        and row["public_repos"] >= 10
        and row["recency_days"] <= 730
    )


def synthetic_profiles(n: int, seed: int) -> pd.DataFrame:
    """Heavy-tailed, loosely correlated profiles resembling public GitHub accounts."""
    rng = np.random.default_rng(seed)
    activity = rng.normal(0, 1, n)  # latent "how active in open source" factor
    account_age_days = rng.integers(30, 5500, n)
    public_repos = np.clip(np.round(np.exp(2.5 + 0.9 * activity + rng.normal(0, 0.6, n))), 0, 600)
    followers = np.round(np.exp(3.5 + 1.6 * activity + rng.normal(0, 0.9, n)))
    stars_received = np.round(np.exp(3.0 + 1.9 * activity + rng.normal(0, 1.0, n)))
    watchers_received = np.round(stars_received * rng.uniform(0.9, 1.0, n))
    language_diversity = np.clip(np.round(1 + 0.25 * public_repos ** 0.6 + rng.normal(0, 1, n)), 0, 25)
    recency_days = np.clip(np.round(np.exp(4.5 - 1.2 * activity + rng.normal(0, 1.2, n))), 0, account_age_days)
    avg_repo_stars = stars_received / np.maximum(public_repos, 1)
    forked_ratio = np.clip(rng.beta(2, 5, n) - 0.05 * activity, 0, 1)
    has_description_ratio = np.clip(rng.beta(4, 3, n) + 0.05 * activity, 0, 1)
    return pd.DataFrame({
        "account_age_days": account_age_days.astype(float),
        "followers": followers,
        "public_repos": public_repos,
        "stars_received": stars_received,
        "language_diversity": language_diversity,
        "recency_days": recency_days.astype(float),
        "avg_repo_stars": avg_repo_stars,
        "forked_ratio": forked_ratio,
        "has_description_ratio": has_description_ratio,
        "watchers_received": watchers_received,
    })


def github_profiles(path: str) -> pd.DataFrame:
    from data5580_hw.services.connectors.github_connector import GitHubConnector, GitHubConnectorError

    connector = GitHubConnector()
    rows = []
    with open(path) as handle:
        usernames = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    for username in usernames:
        try:
            rows.append(connector.extract_features(username))
            print(f"  fetched {username}")
        except GitHubConnectorError as exc:
            print(f"  skipped {username}: {exc}")
    if not rows:
        sys.exit("No profiles fetched.")
    return pd.DataFrame(rows)[FEATURES]


def register(api_url: str, api_key: str, name: str, version: str, run_id: str) -> None:
    headers = {"X-API-Key": api_key} if api_key else {}
    with open("model.pkl", "rb") as handle:
        response = requests.post(
            f"{api_url.rstrip('/')}/models",
            headers=headers,
            files={"file": ("model.pkl", handle)},
            data={
                "name": name,
                "version": version,
                "type": "tree",
                "features": json.dumps(FEATURES),
                "mlflow_run_id": run_id,
            },
            timeout=30,
        )
    print(f"Register {name} v{version}: HTTP {response.status_code} {response.text.strip()}")
    if response.status_code not in (201, 409):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", type=int, default=2000, metavar="N")
    source.add_argument("--usernames", metavar="FILE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="candidate-scorer")
    parser.add_argument("--version", default="1")
    parser.add_argument("--register", metavar="API_URL", help="e.g. http://localhost:5000")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("candidate-scoring")

    data = github_profiles(args.usernames) if args.usernames else synthetic_profiles(args.synthetic, args.seed)
    data["hired"] = data.apply(hired_label, axis=1)
    data.to_csv("training_data.csv", index=False)
    print(f"{len(data)} profiles, {data['hired'].mean():.1%} labeled hired")

    stratify = data["hired"] if data["hired"].nunique() > 1 and data["hired"].value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["hired"], test_size=0.2, random_state=args.seed, stratify=stratify
    )
    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": args.seed}
    model = GradientBoostingRegressor(**params).fit(x_train, y_train)
    predictions = model.predict(x_test)

    with mlflow.start_run() as run:
        mlflow.log_params({**params, "data_source": "github" if args.usernames else "synthetic", "n_rows": len(data)})
        mlflow.log_metrics({
            "mse": mean_squared_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "hired_rate": float(data["hired"].mean()),
        })
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_train.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    with open("model.pkl", "wb") as handle:
        pickle.dump(model, handle)
    with open("model_run_id.txt", "w") as handle:
        handle.write(run_id)
    print(f"MLflow run {run_id} ({tracking_uri}); test MSE {mean_squared_error(y_test, predictions):.4f}")

    if args.register:
        register(args.register, args.api_key, args.name, args.version, run_id)


if __name__ == "__main__":
    main()
