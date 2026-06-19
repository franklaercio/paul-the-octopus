"""Orquestra os notebooks do pipeline em ordem e valida as entradas e os artefatos.

Ordem executável: 01_features -> 02_train -> 03_predict (00_eda fica de fora).

A lógica científica vive nos notebooks; este script valida os contratos dos CSVs
de entrada, executa os notebooks em ordem, valida os artefatos de saída de cada
etapa (à medida que são implementadas) e salva os notebooks executados em
artifacts/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from scripts.validate_data import ROOT, ValidationError, validate_repository, validate_schedule

NOTEBOOKS_DIR = ROOT / "notebooks"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"

# Notebooks que compõem o pipeline executável, na ordem. 00_eda fica de fora.
PIPELINE_NOTEBOOKS = (
    "01_features.ipynb",
    "02_train.ipynb",
    "03_predict.ipynb",
)

# Contrato de features.parquet (ordem e tipos conforme docs/features/decisoes.md §10).
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
FEATURES_COLUMNS = (
    "split",
    "match_no",
    "date",
    "home",
    "away",
    "is_neutral",
    "confed_home",
    "confed_away",
    "elo_home",
    "elo_away",
    "elo_diff",
    "rank_home",
    "rank_away",
    "rank_diff",
    "points_diff",
    "form_pts_home",
    "form_pts_away",
    "form_gf_home",
    "form_gf_away",
    "form_ga_home",
    "form_ga_away",
    "rest_days_home",
    "rest_days_away",
    "h2h_games",
    "h2h_winrate_home",
    "h2h_available",
    "tournament_tier",
    "target_outcome",
    "home_score",
    "away_score",
    "sample_weight",
)
# Colunas de chave que nunca podem ter NaN (qualquer split).
FEATURES_KEY_COLUMNS = ("split", "date", "home", "away", "is_neutral")
# Colunas presentes sse split == "train" (nulas em predict).
FEATURES_TRAIN_ONLY_COLUMNS = ("target_outcome", "home_score", "away_score", "sample_weight")

# Contrato de models/model.joblib (bundle do 02; ver docs/train/README.md §2.1/§10.7).
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_REQUIRED_META = ("feature_names", "classes", "seed", "shrink_alpha", "base_rate")
# Ordem ordinal das classes para o RPS (home -> draw -> away). classes_ do sklearn é
# alfabético (away, draw, home); o 03 mapeia POR NOME via esta lista (nunca por posição).
MODEL_EXPECTED_CLASSES = ["home", "draw", "away"]
# As 22 features que o 02 congela como X (FEATURE_NAMES; docs/train/README.md §8).
MODEL_FEATURE_NAMES = (
    "is_neutral",
    "confed_home",
    "confed_away",
    "elo_home",
    "elo_away",
    "elo_diff",
    "rank_home",
    "rank_away",
    "rank_diff",
    "points_diff",
    "form_pts_home",
    "form_pts_away",
    "form_gf_home",
    "form_gf_away",
    "form_ga_home",
    "form_ga_away",
    "rest_days_home",
    "rest_days_away",
    "h2h_games",
    "h2h_winrate_home",
    "h2h_available",
    "tournament_tier",
)


def proba_in_outcome_order(pipe, X_in, classes=MODEL_EXPECTED_CLASSES) -> np.ndarray:
    """Reordena predict_proba de classes_ (alfabético) para a ordem ordinal do RPS.

    classes_ do sklearn vem de np.unique (away, draw, home); o RPS exige
    (home, draw, away). Mapear POR NOME (nunca por posição) — guardrail R1. A
    MESMA lógica do harness/03; centralizar evita mapeamentos divergentes.
    """
    cols = [list(pipe.classes_).index(o) for o in classes]
    return pipe.predict_proba(X_in)[:, cols]


def apply_shrinkage(proba: np.ndarray, base_rate, alpha: float) -> np.ndarray:
    """Encolhe as probabilidades para a taxa-base: P' = (1-alpha)*P + alpha*base.

    `base_rate` na ordem OUTCOMES (home/draw/away). Com alpha=0 reproduz `proba`
    EXATAMENTE (cru); preserva o simplex (combinação convexa de vetores que somam
    1). A MESMA função do 02/03 — o mesmo ajuste em treino, validação e produção.
    """
    if alpha == 0.0:
        return proba
    base = np.asarray(base_rate, dtype=float).reshape(1, -1)
    return (1.0 - alpha) * proba + alpha * base


def ensemble_proba(proba_clf: np.ndarray, proba_dc: np.ndarray, weight: float) -> np.ndarray:
    """Combina o 1X2 do classificador com o 1X2 implícito de Dixon-Coles.

    P = weight*proba_clf + (1-weight)*proba_dc — combinação convexa de dois
    simplexos, logo um simplex. Com weight=1.0 reproduz o classificador EXATAMENTE
    (retrocompatível: bundles sem ensemble usam weight=1). A MESMA função no 02, no
    03 e na validate_model — fonte única do ensemble. O shrinkage (apply_shrinkage)
    é aplicado DEPOIS, sobre o resultado deste ensemble.
    """
    if weight >= 1.0:
        return proba_clf
    w = float(weight)
    return w * np.asarray(proba_clf, dtype=float) + (1.0 - w) * np.asarray(proba_dc, dtype=float)


# --------------------------------------------------------------------------- #
# Modelo de gols COMPANHEIRO (Poisson + correção Dixon-Coles)
#
# Camada ILUSTRATIVA: NÃO altera as probabilidades 1X2 nem o shrink_alpha. Deriva
# os gols esperados (lambda) de dois regressores de Poisson e monta a matriz de
# placares P(h,a) com o termo de baixos placares de Dixon-Coles, da qual saem
# `xg_home`/`xg_away` e o `placar_provavel`. As funções abaixo são a FONTE ÚNICA
# compartilhada por 02 (sanidade/persistência), 03 (submissão) e validate_model
# (smoke) — centralizar evita derivações divergentes de lambda/placar.
# --------------------------------------------------------------------------- #
# Grade de placares 0..GOALS_MAX por lado (cauda de Poisson além disso é ~0 para os
# lambdas de futebol, ~1-3). O índice 1X2 sai de comparar h vs a nessa grade.
GOALS_MAX = 10
# Faixa válida de rho da correção Dixon-Coles (clamp do MLE; fallback se degenerar).
RHO_BOUNDS = (-0.2, 0.2)
RHO_FALLBACK = -0.13  # coerente com a corr. home/away ~ -0.145 da EDA


def dixon_coles_tau(h, a, lam_home: float, lam_away: float, rho: float) -> np.ndarray:
    """Termo de dependência de baixos placares de Dixon-Coles tau(h,a).

    Reweighta as 4 células (0-0, 0-1, 1-0, 1-1); 1 em qualquer outra. rho<0
    aumenta P(0-0)/P(1-1) e reduz P(1-0)/P(0-1) — corrige o leve déficit de empate
    e a dependência negativa observados na EDA. Vetorizado em h, a (broadcasting).
    """
    h = np.asarray(h)
    a = np.asarray(a)
    tau = np.ones(np.broadcast(h, a).shape, dtype=float)
    tau = np.where((h == 0) & (a == 0), 1.0 - lam_home * lam_away * rho, tau)
    tau = np.where((h == 0) & (a == 1), 1.0 + lam_home * rho, tau)
    tau = np.where((h == 1) & (a == 0), 1.0 + lam_away * rho, tau)
    tau = np.where((h == 1) & (a == 1), 1.0 - rho, tau)
    return tau


def score_matrix(lam_home: float, lam_away: float, rho: float, max_goals: int = GOALS_MAX):
    """Matriz (max_goals+1, max_goals+1) de P(home=h, away=a), normalizada a 1.

    P(h,a) = Poisson(h; lam_home) * Poisson(a; lam_away) * tau(h,a). A diagonal é o
    empate; o triângulo inferior/superior, vitória do mandante/visitante.
    """
    from scipy.stats import poisson

    h = np.arange(max_goals + 1)
    a = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(h, lam_home), poisson.pmf(a, lam_away))
    hh, aa = np.meshgrid(h, a, indexing="ij")
    grid = grid * dixon_coles_tau(hh, aa, lam_home, lam_away, rho)
    total = grid.sum()
    return grid / total if total > 0 else grid


def matrix_to_1x2(matrix: np.ndarray) -> np.ndarray:
    """Soma a matriz de placares na ordem 1X2 (home=h>a, draw=h=a, away=h<a)."""
    n = matrix.shape[0]
    hh, aa = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.array(
        [matrix[hh > aa].sum(), matrix[hh == aa].sum(), matrix[hh < aa].sum()]
    )


def _outcome_mask(matrix: np.ndarray, outcome: str) -> np.ndarray:
    """Máscara booleana das células da matriz compatíveis com o resultado 1X2."""
    n = matrix.shape[0]
    hh, aa = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if outcome == "home":
        return hh > aa
    if outcome == "away":
        return hh < aa
    return hh == aa  # draw


def most_likely_score(matrix: np.ndarray, outcome: str) -> tuple[int, int]:
    """Placar argmax da matriz RESTRITO à região do resultado 1X2 previsto.

    Consistência obrigatória: o placar nunca contradiz a coluna 1X2 (home: h>a;
    draw: h=a; away: h<a). Desempate determinístico: argmax achata em ordem C
    (h cresce primeiro), então o menor (h, a) vence empates de probabilidade.
    """
    mask = _outcome_mask(matrix, outcome)
    masked = np.where(mask, matrix, -1.0)
    h, a = np.unravel_index(int(np.argmax(masked)), matrix.shape)
    return int(h), int(a)


def predict_goals(goals_bundle: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """Lambda (gols esperados) de mandante e visitante para as linhas de X.

    Usa os dois pipelines de Poisson do bundle, na ordem de feature_names. Clampa
    em [1e-6, inf) para a matriz de Poisson ser sempre válida (lambda>0). A MESMA
    derivação no 02, no 03 e na validate_model — sem skew.
    """
    feats = list(goals_bundle["feature_names"])
    lam_home = np.clip(goals_bundle["home_model"].predict(X[feats]), 1e-6, None)
    lam_away = np.clip(goals_bundle["away_model"].predict(X[feats]), 1e-6, None)
    return lam_home, lam_away


def scores_from_lambdas(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float, outcomes
) -> tuple[np.ndarray, list[str]]:
    """xg arredondado (1 casa) e `placar_provavel` consistente com o 1X2 previsto.

    `outcomes` é a lista de resultados 1X2 previstos (1 por jogo, ordem OUTCOMES)
    a que o placar fica restrito (most_likely_score). Devolve (xg (n,2), placares).
    """
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    xg = np.column_stack([np.round(lam_home, 1), np.round(lam_away, 1)])
    placares = []
    for i, outcome in enumerate(outcomes):
        matrix = score_matrix(lam_home[i], lam_away[i], rho)
        h, a = most_likely_score(matrix, outcome)
        placares.append(f"{h}-{a}")
    return xg, placares


def implied_1x2_from_goals(goals_bundle: dict, X) -> np.ndarray:
    """1X2 implícito (home, draw, away) pela matriz de placares Dixon-Coles.

    Para cada linha de X: deriva lambda dos regressores de gols, monta a matriz de
    placares e soma as regiões (home>away, empate, home<away). É o componente de
    gols usado como PREDITOR 1X2 no ensemble — fonte única reusada pelo 02 (tunagem),
    pelo 03 (produção) e pela validate_model (smoke).
    """
    lam_home, lam_away = predict_goals(goals_bundle, X)
    rho = float(goals_bundle["rho"])
    proba = np.empty((len(lam_home), 3))
    for i in range(len(lam_home)):
        proba[i] = matrix_to_1x2(score_matrix(lam_home[i], lam_away[i], rho))
    return proba


def execute_notebook(path: Path, output_dir: Path, timeout: int) -> Path:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = output_dir / f"{path.stem}.executed.ipynb"
    nbformat.write(notebook, executed)
    return executed


def validate_features(path: Path = FEATURES_PATH, schedule_dir: Path = RAW_DATA_DIR) -> int:
    """Valida o artefato features.parquet contra o contrato da §10 da spec.

    Garante: colunas e ordem exatas; sem NaN nas chaves; exatamente 72 linhas
    predict cobrindo match 1..72 do calendário; alvo/placar/peso presentes sse
    split == "train".
    """
    if not path.is_file():
        raise ValidationError(f"Artefato ausente: {path}")

    features = pd.read_parquet(path)

    if tuple(features.columns) != FEATURES_COLUMNS:
        missing = set(FEATURES_COLUMNS) - set(features.columns)
        extra = set(features.columns) - set(FEATURES_COLUMNS)
        detail = []
        if missing:
            detail.append(f"ausentes: {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"inesperadas: {', '.join(sorted(extra))}")
        if not detail:
            detail.append("ordem das colunas diverge do contrato")
        raise ValidationError(f"features.parquet: colunas ({'; '.join(detail)})")

    valid_splits = {"train", "predict"}
    if not set(features["split"].dropna().unique()).issubset(valid_splits):
        raise ValidationError("features.parquet: split deve ser 'train' ou 'predict'")

    for col in FEATURES_KEY_COLUMNS:
        if features[col].isna().any():
            raise ValidationError(f"features.parquet: NaN na coluna-chave '{col}'")

    predict = features[features["split"] == "predict"]
    n_predict = len(predict)
    n_schedule = validate_schedule(schedule_dir / "matches-schedule.csv")
    if n_predict != n_schedule:
        raise ValidationError(
            f"features.parquet: esperado {n_schedule} linhas predict, obtido {n_predict}"
        )
    match_numbers = sorted(int(x) for x in predict["match_no"].dropna().tolist())
    if match_numbers != list(range(1, n_schedule + 1)):
        raise ValidationError(
            "features.parquet: linhas predict devem cobrir match 1.."
            f"{n_schedule} sem buracos"
        )

    train = features[features["split"] == "train"]
    for col in FEATURES_TRAIN_ONLY_COLUMNS:
        if not train[col].notna().all():
            raise ValidationError(f"features.parquet: '{col}' deve estar presente em todo train")
        if not predict[col].isna().all():
            raise ValidationError(f"features.parquet: '{col}' deve ser nulo em predict")

    return len(features)


def validate_model(model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH) -> int:
    """Valida o artefato models/model.joblib contra o contrato da §2.1/§10.8 da spec.

    Garante, **sem** o gabarito: desserializa; é um dict com 'pipeline' e 'meta';
    meta tem feature_names/classes/seed/shrink_alpha/base_rate; classes ==
    ["home","draw","away"] (ordem do RPS) e casa com pipeline.classes_;
    feature_names é a lista esperada e está contida nas colunas do parquet;
    shrink_alpha em [0,1] e base_rate é um vetor de 3 não-negativos que soma ~1;
    e o smoke de predição nas 72 linhas predict — mapeado por nome e JÁ com o
    shrinkage aplicado (como o 03) — devolve (72, 3) sem NaN, em [0, 1] e somando
    ~1, e com alpha=0 reproduz o cru (garante que o 03 vai funcionar).

    Retorna o número de linhas previstas (72), no estilo de validate_features.
    """
    if not model_path.is_file():
        raise ValidationError(f"Artefato ausente: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or {"pipeline", "meta"} - bundle.keys():
        raise ValidationError("model.joblib: esperado dict com 'pipeline' e 'meta'")
    pipe, meta = bundle["pipeline"], bundle["meta"]

    for key in MODEL_REQUIRED_META:
        if key not in meta:
            raise ValidationError(f"model.joblib: meta sem '{key}'")
    if list(meta["classes"]) != MODEL_EXPECTED_CLASSES:
        raise ValidationError(f"model.joblib: classes != {MODEL_EXPECTED_CLASSES}")
    if set(meta["classes"]) != set(pipe.classes_):
        raise ValidationError("model.joblib: meta['classes'] != pipeline.classes_")

    feats = list(meta["feature_names"])
    if tuple(feats) != MODEL_FEATURE_NAMES:
        raise ValidationError("model.joblib: feature_names diverge do contrato (§8)")

    alpha = float(meta["shrink_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValidationError(f"model.joblib: shrink_alpha fora de [0,1]: {alpha}")
    base_rate = np.asarray(meta["base_rate"], dtype=float)
    if base_rate.shape != (3,) or (base_rate < 0).any() or not np.isclose(base_rate.sum(), 1.0):
        raise ValidationError("model.joblib: base_rate inválido (esperado 3 não-neg. somando 1)")

    features = pd.read_parquet(features_path)
    missing = set(feats) - set(features.columns)
    if missing:
        raise ValidationError(
            f"model.joblib: feature_names fora do parquet: {sorted(missing)}"
        )

    predict = features[features["split"] == "predict"]
    # Smoke como o 03: mapeia por nome para a ordem do RPS, combina com o 1X2
    # implícito de Dixon-Coles (ensemble, se configurado) e aplica o shrinkage.
    proba_clf = proba_in_outcome_order(pipe, predict[feats])
    ens_weight = float(meta.get("ensemble_weight", 1.0))
    if not 0.0 <= ens_weight <= 1.0:
        raise ValidationError(f"model.joblib: ensemble_weight fora de [0,1]: {ens_weight}")
    if ens_weight < 1.0:
        if "goals" not in bundle:
            raise ValidationError("model.joblib: ensemble_weight<1 exige o componente 'goals'")
        proba_dc = implied_1x2_from_goals(bundle["goals"], predict)
        proba_1x2 = ensemble_proba(proba_clf, proba_dc, ens_weight)
    else:
        proba_1x2 = proba_clf
    proba = apply_shrinkage(proba_1x2, base_rate, alpha)
    if proba.shape != (len(predict), 3):
        raise ValidationError(f"model.joblib: predict_proba shape {proba.shape}")
    if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
        raise ValidationError("model.joblib: probabilidades inválidas (NaN/fora de [0,1])")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValidationError("model.joblib: probabilidades não somam 1")
    if alpha == 0.0 and not np.array_equal(proba, proba_1x2):
        raise ValidationError("model.joblib: shrink_alpha=0 deveria reproduzir o 1X2 sem shrinkage")

    # Componente de gols COMPANHEIRO: lambda finito/positivo nos 72 e placar
    # consistente com o 1X2 FINAL (a MESMA derivação do 03).
    if "goals" in bundle:
        validate_goals(bundle["goals"], predict, proba)

    return len(predict)


def validate_goals(goals_bundle: dict, predict: pd.DataFrame, proba_1x2: np.ndarray) -> None:
    """Smoke do componente de gols: lambda finito/positivo e placar consistente.

    Garante, sem o gabarito: o bundle de gols tem home_model/away_model/rho/
    feature_names; rho na faixa válida; lambda_home/away finitos e > 0 nos 72 jogos;
    a matriz de placares soma ~1; e o `placar_provavel` cai na região do resultado
    1X2 do classificador (home: h>a; draw: h=a; away: h<a) — nunca contradiz o 1X2.
    """
    for key in ("home_model", "away_model", "rho", "feature_names"):
        if key not in goals_bundle:
            raise ValidationError(f"model.joblib: bundle 'goals' sem '{key}'")
    rho = float(goals_bundle["rho"])
    if not RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]:
        raise ValidationError(f"model.joblib: rho fora de {RHO_BOUNDS}: {rho}")

    lam_home, lam_away = predict_goals(goals_bundle, predict)
    if lam_home.shape != (len(predict),) or lam_away.shape != (len(predict),):
        raise ValidationError("model.joblib: lambda de gols com shape inesperado")
    for lam in (lam_home, lam_away):
        if not np.isfinite(lam).all() or (lam <= 0).any():
            raise ValidationError("model.joblib: lambda de gols não finito ou <= 0")

    outcomes = [MODEL_EXPECTED_CLASSES[i] for i in np.argmax(proba_1x2, axis=1)]
    _, placares = scores_from_lambdas(lam_home, lam_away, rho, outcomes)
    for outcome, placar in zip(outcomes, placares):
        h, a = (int(v) for v in placar.split("-"))
        ok = (h > a) if outcome == "home" else (h < a) if outcome == "away" else (h == a)
        if not ok:
            raise ValidationError(
                f"model.joblib: placar '{placar}' inconsistente com 1X2 '{outcome}'"
            )


# Validações de artefato por notebook (acrescentar conforme cada etapa amadurece).
ARTIFACT_VALIDATORS = {
    "01_features.ipynb": validate_features,
    "02_train.ipynb": validate_model,
}


def run_pipeline(output_dir: Path, timeout: int) -> None:
    validate_repository()
    for name in PIPELINE_NOTEBOOKS:
        path = NOTEBOOKS_DIR / name
        if not path.is_file():
            raise ValidationError(f"Notebook ausente: {path}")
        executed = execute_notebook(path, output_dir, timeout)
        print(f"OK: {name} executado -> {executed.name}")
        validator = ARTIFACT_VALIDATORS.get(name)
        if validator is not None:
            count = validator()
            print(f"OK: artefato de {name} validado ({count} linhas)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa os notebooks do pipeline em ordem e valida as entradas."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Pasta onde gravar os notebooks executados.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Timeout por celula, em segundos.")
    args = parser.parse_args()

    try:
        run_pipeline(args.output_dir.resolve(), args.timeout)
    except (ValidationError, Exception) as exc:
        print(f"ERRO: {exc}")
        return 1

    print("OK: pipeline executado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
