from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scripts.validate_data import ROOT, ValidationError, validate_repository

NOTEBOOK = ROOT / "notebooks" / "paultheoctopus.ipynb"
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS = ROOT / "data" / "results" / "predictions_submission.csv"


def validate_predictions(path: Path, expected_rows: int) -> None:
    if not path.is_file():
        raise ValidationError(f"O pipeline nao gerou {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"home", "home_score", "away_score", "away"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValidationError(
                f"predictions_submission.csv: colunas ausentes: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    if len(rows) != expected_rows:
        raise ValidationError(
            "predictions_submission.csv: quantidade de jogos diferente do calendario "
            f"({len(rows)} != {expected_rows})"
        )


def run_pipeline(output: Path, timeout: int) -> None:
    counts = validate_repository()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))

    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    validate_predictions(PREDICTIONS, counts["matches-schedule.csv"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa o notebook completo e valida a saida.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "paultheoctopus.executed.ipynb",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Timeout por celula, em segundos.")
    args = parser.parse_args()

    try:
        run_pipeline(args.output.resolve(), args.timeout)
    except (ValidationError, Exception) as exc:
        print(f"ERRO: {exc}")
        return 1

    print(f"OK: notebook executado em {args.output.resolve()}")
    print(f"OK: previsoes validadas em {PREDICTIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
