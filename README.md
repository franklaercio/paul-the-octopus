# Paul the Octopus

Data science and machine learning pipeline for predicting the results of the FIFA 2026 World Cup. The project combines match history, FIFA ranking, probability calibration, and a goals model to generate reproducible predictions for the 72 group-stage matches.

![Paul the Octopus](img/paul.png)

> [!IMPORTANT]
> `data/results/predictions_submission.csv` is a generated artifact. Running the full pipeline overwrites this file. Review the prediction diff before committing.

## Current model

- **1X2 result:** P2 classifier ensemble with isotonic calibration and Dixon-Coles.
- **Scoreline:** neutral-field Dixon-Coles matrix, with a goal scale of `1.35`.
- **Competitive draws:** flexible margin `DRAW_MARGIN_PROD=0.08`.
- **Validation:** temporal splits and 2022 World Cup hold-out, preserved against leakage.

The experimental decisions and metrics are documented in
[docs/AVALIACAO-PREVISOES-2026.md](docs/AVALIACAO-PREVISOES-2026.md).

## Quick start

Requires Python 3.10 or higher.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m pytest
python -m scripts.run_pipeline
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m pytest
python -m scripts.run_pipeline
```

## Structure

```text
data/raw/                 Input CSVs
data/results/             Generated predictions
notebooks/paultheoctopus.ipynb  Main analysis, training, and inference pipeline
scripts/                  Data validation and automated execution
tests/                    Input contract tests
docs/                     Decisions, evaluations, and technical documentation
artifacts/                Executed notebook, generated and not versioned
```

To install only the runtime environment, without development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execution

Validate the input CSVs:

```bash
python -m scripts.validate_data
```

Run the entire notebook and validate the predictions:

```bash
python -m scripts.run_pipeline
```

The command generates:

| File | Description |
|---|---|
| `data/results/predictions_submission.csv` | Direct model output, with selections and predicted scoreline |
| `artifacts/paultheoctopus.executed.ipynb` | Executed notebook, metrics, and execution diagnostics |

The file `data/results/schedule_with_predictions.csv` keeps the schedule enriched with the predicted scoreline and result for human consumption. It must remain consistent with the submission when both represent the same model run.

To work interactively:

```bash
jupyter lab notebooks/paultheoctopus.ipynb
```

## Quality

```bash
python -m ruff check scripts tests
python -m pytest
```

The workflow in `.github/workflows/pipeline.yml` installs the environment, validates the data, runs the tests, and executes the full notebook on pushes and pull requests.

## Data

Inputs are stored in `data/raw/` and results in `data/results/`. The pipeline uses local CSVs and does not depend on GCP or a database.

| Input | Purpose |
|---|---|
| `historical-results.csv` | Match history used for training and temporal features |
| `ranking.csv` | FIFA ranking history available before each match |
| `matches-schedule.csv` | Schedule of the 72 matches planned for 2026 |

The schedule uses Brasília date and time: `date` in `DD/MM/YYYY`, `time_brasilia` in `HH:MM`, and `timezone=GMT-3`.

## Reproducibility

Scientific changes should be made in `notebooks/paultheoctopus.ipynb`. The scripts only orchestrate execution and validate contracts; they do not duplicate the model logic.

Before accepting new predictions:

1. Run `python -m scripts.validate_data`.
2. Run `python -m pytest`.
3. Run `python -m scripts.run_pipeline`.
4. Review the executed notebook metrics and the diff in `data/results/`.

See [docs/PIPELINE.md](docs/PIPELINE.md) for contracts, steps, and troubleshooting.

## License

MIT. Author: Frank Laércio.
