# CandidateScoringML

A system that scores software-engineering candidates from their public GitHub activity. It learns from hiring-outcome feedback over time, explains every score with SHAP, and has a local LLM write a plain-language rationale grounded in similar past candidates.

Under the hood it is a general model-serving API: the candidate-scoring layer turns a GitHub username into a feature vector and hands it to a model-agnostic prediction pipeline that knows nothing about GitHub or hiring.

## How a candidate is scored

```
POST /candidates/<github_username>/score
        │
        ▼
GitHub connector ──► 10-feature behavioral vector
        │
        ▼
Model inference (registered scikit-learn model) ──► raw fit score
        │
        ▼
Explore/exploit adjustment (ε decays as outcome labels accumulate)
        │
        ├──► SHAP explanation        (why the model scored it this way)
        ├──► UMAP embedding          (where the candidate sits among past candidates)
        ├──► MLflow artifact logging
        ▼
Stored prediction ──► Arize monitoring
        │
        ▼
PATCH /predictions/<id>/actual   ← hiring outcome closes the loop
GET   /predictions/<id>/explainer → LLM rationale using nearest neighbors
```

## Design highlights

### Learning from outcome feedback (cold-start explore/exploit)
A new model has no evidence about real hiring outcomes, so trusting its scores immediately would keep surfacing the same kind of candidate. Scoring uses an epsilon-greedy policy whose exploration rate depends on how many outcomes have been recorded for that model:

```
ε = max(0.05, 0.8 · exp(−0.05 · n_labeled_outcomes))
```

With no feedback, about 80% of scores are randomized to diversify who gets surfaced. Each outcome recorded through the feedback endpoint lowers ε, and after roughly 60 labeled outcomes the system is close to pure exploitation (ε ≈ 0.05). The current ε is returned with every score, so the caller knows how much to trust it.

### LLM explanations grounded in retrieval
`GET /predictions/<id>/explainer` finds the most similar past candidates by cosine similarity in UMAP embedding space. It then prompts a locally hosted LLM (Qwen2.5-14B through Ollama) with the candidate's score, behavioral signals, role context and those neighbors. The model writes a short rationale for a hiring manager: what the score means, and how this candidate differs from the most similar ones. Inference stays local, so candidate data is never sent to a third-party API.

### Feature engineering from GitHub
The connector calls the GitHub REST API and builds a 10-feature vector:
- account age, followers, public repos
- total stars received and watchers
- language diversity
- days since last push
- average stars per repo
- share of forked repos
- share of repos with a description

Results are cached for an hour, and the connector handles unknown users and rate limits. Signal sources are pluggable: a new source is a module that emits the same feature dictionary, and nothing downstream changes.

### Model and labels
The scoring model is a scikit-learn `GradientBoostingRegressor`. [`scripts/train_model.py`](scripts/train_model.py) trains it, logs it to MLflow (hyperparameters, plus MSE and R² on a held-out split), and can register it with the API in the same step. Profiles are labeled with a deliberately strict rule: ≥500 followers, ≥300 stars received, ≥10 repos, and active in the last two years.

There are two data sources:
- By default, the script generates synthetic GitHub-like profiles, so it runs without network access and without attaching labels to real people.
- `--usernames FILE` fetches real public profiles through the same connector the API uses.

The first version was trained on about 50 hand-picked accounts. See [Limitations](#limitations-and-what-id-change) for what this labeling scheme means.

The model is trained in the same environment that serves it (dependencies are pinned), because scikit-learn pickles don't load across versions.

### Explainability
Every prediction stores per-feature SHAP values. The explainer is chosen from the model family (tree, linear, deep, or a kernel fallback) and cached per model version.

### Serving
- Model registry with MLflow run linkage.
- Synchronous and Celery/Redis async inference.
- PostgreSQL or SQLite storage.
- Prometheus request metrics and Arize model monitoring.
- Optional API-key auth.
- Docker Compose stack: API, Celery worker, Postgres, Redis and an MLflow tracking server.
- 303 tests, with every external service (GitHub, MLflow, Arize, Ollama) mocked.

## Limitations and what I'd change

- **The model learns the labeling rule, not hiring quality.** The labels are a deterministic function of the features, so the model mostly recovers that rule. Its scores reflect the labeling policy rather than real hiring outcomes. The feedback endpoint exists to replace those weak labels with real outcomes. The next step would be retraining on those outcomes and evaluating on a held-out set of real decisions.
- **The explored score and its explanation can disagree.** When exploration randomizes a score, the randomized value is what gets returned, but the SHAP values and the LLM rationale describe the model's actual output. I'd return both scores, and explore over which candidates are ranked and surfaced rather than over the score itself. Thompson sampling or UCB would also be better-grounded choices than a fixed decay schedule.
- **Feedback is selection-biased.** Outcomes only exist for candidates someone chose to advance. Without correction (e.g. logging the probability each candidate was surfaced and reweighting by it), the model would retrain on a skewed sample.
- **GitHub activity is a biased proxy.** It favors people with the time and freedom to work in public. Any real use would need a bias evaluation across groups and a human making the decision.
- **The embedding projection is fitted lazily.** UMAP is fitted on the first prediction a model sees and then reused for every later prediction. Coordinates are only a rough position, and when a projection degenerates they fall back to 0.0. Fitting the projection on a reference set at registration time would make nearest-neighbor retrieval meaningful.
- **Explanation faithfulness is not measured.** The LLM rationale isn't checked against the SHAP attributions. An evaluation that scores whether the claims in each rationale match the actual feature attributions would be the next thing to build.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/candidates/<github_username>/score` | Score a candidate. Body: `model_name`, `model_version`, `role_context`, optional `umap` config |
| POST | `/models` | Register a model: multipart `file` (pickle), `name`, `version`, `features`, `mlflow_run_id`, optional `type` |
| GET | `/models` | List registered models |
| POST | `/predictions/model/<name>/version/<version>` | Generic prediction from a feature dict |
| GET | `/predictions/<id>` | Fetch a stored prediction |
| GET | `/predictions/<id>/explainer` | LLM rationale using nearest neighbors (needs Ollama) |
| PATCH | `/predictions/<id>/actual` | Record an outcome for one prediction |
| PATCH | `/predictions/actuals` | Record outcomes in bulk: `{"updates": [{"prediction_id", "actual"}]}` |
| POST | `/tasks/predictions/model/<name>/version/<version>` | Queue an async prediction |
| GET | `/tasks/<task_id>` | Async task status and result |
| GET | `/metrics` | Prometheus metrics |

## Quickstart (Docker)

```bash
cp .env.example .env                  # change API_PORT etc. if a port is taken
docker compose up -d --build          # API :5000, MLflow UI :5001, Postgres, Redis, worker

# Train inside the API container (same library versions as serving) and register the model
docker compose exec api python scripts/train_model.py --register http://localhost:5000

# Score a candidate
curl -X POST localhost:5000/candidates/<github_username>/score \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "candidate-scorer", "model_version": "1", "role_context": "ML engineer"}'

# Record the outcome; exploration decays as outcomes accumulate
curl -X PATCH localhost:5000/predictions/<prediction_id>/actual \
  -H 'Content-Type: application/json' -d '{"actual": 1}'
```

On macOS, AirPlay Receiver usually holds port 5000. Set `API_PORT=5050` in `.env` and use that port instead.

**LLM explainer:** run [Ollama](https://ollama.com) on the host with `ollama pull qwen2.5:14b`. Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`, then call `GET /predictions/<id>/explainer`. Loading the model the first time can take a minute; `OLLAMA_TIMEOUT` controls how long the API waits.

## Local development

Requires Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                    # 303 tests, no network or services needed

mlflow server --port 5001 &               # tracking server
export MLFLOW_TRACKING_URI=http://localhost:5001 ARIZE_ENABLED=false
flask --app data5580_hw.app run --port 5050   # SQLite by default; tables reset on each start
python scripts/train_model.py --register http://localhost:5050

# Async endpoints (/tasks) also need Redis and a worker:
celery -A data5580_hw.worker worker --loglevel=info
```

All settings are environment variables, listed with defaults in [`.env.example`](.env.example). `GITHUB_TOKEN` raises the GitHub API limit from 60 to 5,000 requests per hour. When `API_KEY` is set, every request except `/` and `/metrics` needs a matching `X-API-Key` header.

## Tech stack

Python, Flask, SQLAlchemy, Pydantic, scikit-learn, SHAP, UMAP, NumPy, pandas, MLflow, Ollama (Qwen2.5), GitHub REST API, Celery, Redis, PostgreSQL, Prometheus, Arize, Docker, pytest, GitHub Actions.
