from pathlib import Path

import pytest

from scripts.validate_data import (
    RAW_DATA_DIR,
    ValidationError,
    validate_repository,
    validate_schedule,
)


def test_repository_inputs_are_valid() -> None:
    counts = validate_repository(RAW_DATA_DIR)

    assert counts["historical-results.csv"] > 0
    assert counts["ranking.csv"] > 0
    assert counts["matches-schedule.csv"] == 72


def test_schedule_rejects_invalid_timezone(tmp_path: Path) -> None:
    schedule = tmp_path / "matches-schedule.csv"
    schedule.write_text(
        "match,date,time_brasilia,timezone,home,away,phase\n"
        "1,11/06/2026,16:00,UTC,Mexico,South Africa,group matches\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="GMT-3"):
        validate_schedule(schedule)
