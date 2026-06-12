from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"

REQUIRED_COLUMNS = {
    "historical-results.csv": {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    },
    "ranking.csv": {
        "rank",
        "country_full",
        "country_abrv",
        "total_points",
        "previous_points",
        "rank_change",
        "confederation",
        "rank_date",
    },
    "matches-schedule.csv": {
        "match",
        "date",
        "time_brasilia",
        "timezone",
        "home",
        "away",
        "phase",
    },
}


class ValidationError(ValueError):
    """Raised when an input file violates the pipeline contract."""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValidationError(f"Arquivo ausente: {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.get(path.name, set()) - columns
        if missing:
            raise ValidationError(
                f"{path.name}: colunas ausentes: {', '.join(sorted(missing))}"
            )
        return list(reader)


def validate_schedule(path: Path) -> int:
    rows = read_csv(path)
    if not rows:
        raise ValidationError("matches-schedule.csv esta vazio")

    match_numbers: list[int] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            match_numbers.append(int(row["match"]))
            datetime.strptime(row["date"], "%d/%m/%Y")
        except ValueError as exc:
            raise ValidationError(
                f"matches-schedule.csv:{line_number}: numero ou data invalida"
            ) from exc

        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", row["time_brasilia"]):
            raise ValidationError(
                f"matches-schedule.csv:{line_number}: horario invalido"
            )
        if row["timezone"] != "GMT-3":
            raise ValidationError(
                f"matches-schedule.csv:{line_number}: timezone deve ser GMT-3"
            )
        if not row["home"].strip() or not row["away"].strip():
            raise ValidationError(
                f"matches-schedule.csv:{line_number}: selecao nao informada"
            )

    expected = list(range(1, len(rows) + 1))
    if sorted(match_numbers) != expected:
        raise ValidationError(
            "matches-schedule.csv: numeros de partida devem ser unicos e sequenciais"
        )

    return len(rows)


def validate_repository(files_dir: Path = RAW_DATA_DIR) -> dict[str, int]:
    counts = {
        "historical-results.csv": len(read_csv(files_dir / "historical-results.csv")),
        "ranking.csv": len(read_csv(files_dir / "ranking.csv")),
        "matches-schedule.csv": validate_schedule(files_dir / "matches-schedule.csv"),
    }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida os CSVs de entrada do pipeline.")
    parser.add_argument("--files-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    try:
        counts = validate_repository(args.files_dir.resolve())
    except ValidationError as exc:
        print(f"ERRO: {exc}")
        return 1

    for name, count in counts.items():
        print(f"OK: {name} ({count} registros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
