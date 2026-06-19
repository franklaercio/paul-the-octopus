"""Testes do artefato de features (data/processed/features.parquet).

Cobre o contrato (schema/colunas/contagem, alvo só em train) e um smoke test de
vazamento temporal: recomputa Elo, forma e H2H "à mão" para uma partida histórica
usando apenas jogos anteriores e confere que batem com o artefato (nenhuma feature
enxerga o próprio jogo ou o futuro). Também confirma que o notebook não lê o
gabarito (worldcup-2026-results.csv).

Os testes são auto-suficientes: se o features.parquet ainda não existir, a fixture
o gera executando o notebook 01 (o CI roda pytest antes de run_pipeline).
"""
from __future__ import annotations

import sys
import unicodedata

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient

from scripts.run_pipeline import (
    FEATURES_COLUMNS,
    FEATURES_KEY_COLUMNS,
    FEATURES_TRAIN_ONLY_COLUMNS,
    validate_features,
)
from scripts.validate_data import ROOT

NOTEBOOK_01 = ROOT / "notebooks" / "01_features.ipynb"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
HISTORICAL_CSV = ROOT / "data" / "raw" / "historical-results.csv"

ELO_INICIAL = 1500.0
ELO_K = 40.0
ELO_HFA = 65.0
FORM_N = 5

# Mesma fonte de nomes do notebook (skill avaliar-previsoes), p/ o brute-force casar.
_SKILL = ROOT / ".claude" / "skills" / "avaliar-previsoes" / "scripts"
if str(_SKILL) not in sys.path:
    sys.path.insert(0, str(_SKILL))
from score_predictions import DEFAULT_ALIASES  # noqa: E402


def _norm(name: object) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _canon(name: object) -> str:
    key = _norm(name)
    return DEFAULT_ALIASES.get(key, key)


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """features.parquet como DataFrame; gera o artefato se ainda não existir."""
    if not FEATURES_PATH.is_file():
        notebook = nbformat.read(NOTEBOOK_01, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cwd=str(ROOT))
    return pd.read_parquet(FEATURES_PATH)


# --------------------------------------------------------------------------- #
# Contrato / schema
# --------------------------------------------------------------------------- #
def test_columns_exact_order(features_df: pd.DataFrame) -> None:
    assert tuple(features_df.columns) == FEATURES_COLUMNS


def test_validate_features_passes() -> None:
    # Reaproveita a validação canônica do run_pipeline (schema, 72 predict, etc.).
    assert validate_features() > 0


def test_split_values(features_df: pd.DataFrame) -> None:
    assert set(features_df["split"].unique()) == {"train", "predict"}


def test_exactly_72_predict_rows_covering_calendar(features_df: pd.DataFrame) -> None:
    predict = features_df[features_df["split"] == "predict"]
    assert len(predict) == 72
    assert sorted(int(x) for x in predict["match_no"].dropna()) == list(range(1, 73))


def test_no_nan_in_key_columns(features_df: pd.DataFrame) -> None:
    for col in FEATURES_KEY_COLUMNS:
        assert features_df[col].notna().all(), f"NaN em chave {col}"


def test_target_present_iff_train(features_df: pd.DataFrame) -> None:
    train = features_df[features_df["split"] == "train"]
    predict = features_df[features_df["split"] == "predict"]
    for col in FEATURES_TRAIN_ONLY_COLUMNS:
        assert train[col].notna().all(), f"{col} ausente em train"
        assert predict[col].isna().all(), f"{col} presente em predict"


def test_target_outcome_domain(features_df: pd.DataFrame) -> None:
    train = features_df[features_df["split"] == "train"]
    assert set(train["target_outcome"].unique()) <= {"home", "draw", "away"}


def test_predict_strength_features_complete(features_df: pd.DataFrame) -> None:
    # Os 72 jogos de 2026 nunca ficam sem força/contexto (Elo, ranking, forma, rest).
    predict = features_df[features_df["split"] == "predict"]
    required = ("elo_home", "elo_away", "rank_home", "rank_away", "form_pts_home", "rest_days_home")
    for col in required:
        assert predict[col].notna().all(), f"{col} NaN em predict"


def test_h2h_winrate_in_unit_interval(features_df: pd.DataFrame) -> None:
    assert features_df["h2h_winrate_home"].between(0.0, 1.0).all()


def test_tournament_tier_range(features_df: pd.DataFrame) -> None:
    assert features_df["tournament_tier"].between(0, 5).all()
    # Em 2026 todos os jogos são Copa do Mundo (tier 5).
    predict = features_df[features_df["split"] == "predict"]
    assert (predict["tournament_tier"] == 5).all()


# --------------------------------------------------------------------------- #
# Anti-vazamento
# --------------------------------------------------------------------------- #
def test_notebook_does_not_read_ground_truth() -> None:
    # Nenhuma menção ao gabarito (worldcup-2026-results) em nenhuma célula de código,
    # o que cobre qualquer forma de leitura (read_csv/read_parquet/open) do arquivo.
    notebook = nbformat.read(NOTEBOOK_01, as_version=4)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "worldcup-2026-results" not in code.lower(), "notebook referencia o gabarito"


def test_predict_elo_is_pretournament_snapshot(features_df: pd.DataFrame) -> None:
    # Cada seleção tem o MESMO Elo nos seus 3 jogos de 2026 (não realimenta o Mundial).
    predict = features_df[features_df["split"] == "predict"]
    assert predict.groupby("home")["elo_home"].nunique().max() == 1


def _load_clean_history() -> pd.DataFrame:
    h = pd.read_csv(HISTORICAL_CSV, parse_dates=["date"])
    h["home"] = h["home_team"].map(_canon)
    h["away"] = h["away_team"].map(_canon)
    # mesmo dedupe da spec (remove ambas as linhas do par duplicado)
    h = h[~h.duplicated(subset=["date", "home", "away"], keep=False)].copy()
    return h.sort_values(["date", "home", "away"], kind="mergesort").reset_index(drop=True)


def _elo_bruteforce(history: pd.DataFrame, target_idx: int) -> tuple[float, float]:
    """Elo pré-jogo dos dois times na partida target_idx, varrendo só o passado."""
    ratings: dict[str, float] = {}
    for i in range(len(history)):
        row = history.iloc[i]
        rh = ratings.get(row.home, ELO_INICIAL)
        ra = ratings.get(row.away, ELO_INICIAL)
        if i == target_idx:
            return rh, ra
        dr = (rh + (0.0 if row.neutral else ELO_HFA)) - ra
        e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        gd = abs(int(row.home_score) - int(row.away_score))
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        if row.home_score > row.away_score:
            s_home = 1.0
        elif row.home_score == row.away_score:
            s_home = 0.5
        else:
            s_home = 0.0
        ratings[row.home] = rh + ELO_K * g * (s_home - e_home)
        ratings[row.away] = ra + ELO_K * g * ((1.0 - s_home) - (1.0 - e_home))
    raise AssertionError("target_idx fora do histórico")


def _form_pts_bruteforce(history: pd.DataFrame, target_idx: int, team: str) -> float:
    prev = history.iloc[:target_idx]
    games = prev[(prev.home == team) | (prev.away == team)].tail(FORM_N)
    pts = []
    for _, r in games.iterrows():
        gf, ga = (r.home_score, r.away_score) if r.home == team else (r.away_score, r.home_score)
        pts.append(3 if gf > ga else (1 if gf == ga else 0))
    return float(np.mean(pts)) if pts else float("nan")


def _h2h_bruteforce(history: pd.DataFrame, target_idx: int) -> tuple[int, float]:
    target = history.iloc[target_idx]
    prev = history.iloc[:target_idx]
    pair = prev[
        ((prev.home == target.home) & (prev.away == target.away))
        | ((prev.home == target.away) & (prev.away == target.home))
    ]
    n = len(pair)
    if n == 0:
        return 0, 0.5
    wins = 0
    for _, r in pair.iterrows():
        if r.home == target.home:
            gf, ga = r.home_score, r.away_score
        else:
            gf, ga = r.away_score, r.home_score
        wins += int(gf > ga)
    return n, wins / n


def test_no_leak_elo_form_h2h_match_bruteforce(features_df: pd.DataFrame) -> None:
    """Smoke test de vazamento: Elo/forma/H2H do artefato == recomputo só-passado.

    Pega uma partida histórica determinística e confere que as features registradas
    usam exclusivamente jogos anteriores (não enxergam o próprio jogo nem o futuro).
    """
    history = _load_clean_history()
    target_idx = 40000  # determinístico; um jogo "no meio" da história
    target = history.iloc[target_idx]

    elo_home_bf, elo_away_bf = _elo_bruteforce(history, target_idx)
    form_pts_bf = _form_pts_bruteforce(history, target_idx, target.home)
    h2h_games_bf, h2h_wr_bf = _h2h_bruteforce(history, target_idx)

    row = features_df[
        (features_df["split"] == "train")
        & (features_df["home"] == target.home)
        & (features_df["away"] == target.away)
        & (features_df["date"] == pd.Timestamp(target.date).normalize())
    ]
    assert len(row) == 1, "partição alvo não encontrada/única no artefato"
    row = row.iloc[0]

    assert row["elo_home"] == pytest.approx(elo_home_bf, abs=1e-6)
    assert row["elo_away"] == pytest.approx(elo_away_bf, abs=1e-6)
    assert row["form_pts_home"] == pytest.approx(form_pts_bf, abs=1e-9)
    assert int(row["h2h_games"]) == h2h_games_bf
    assert row["h2h_winrate_home"] == pytest.approx(h2h_wr_bf, abs=1e-9)
