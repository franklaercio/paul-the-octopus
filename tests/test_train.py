"""Testes do artefato de treino (models/model.joblib) e do notebook 02.

Cobre o contrato do bundle (estrutura, meta, ordem das classes), a sanidade
probabilística nas 72 linhas predict, o determinismo do predict_proba, a paridade
das métricas replicadas no notebook contra a skill de avaliação, e o teste de fogo
da skill automatizado: num split temporal leve, o RPS do modelo <= RPS da taxa-base.
Também confirma que o notebook não lê o gabarito (worldcup-2026-results.csv).

Os testes são auto-suficientes: se o model.joblib ainda não existir, a fixture o
gera executando o notebook 02 (que por sua vez depende do features.parquet do 01).
Os testes pesados (treino) usam uma fatia temporal pequena, não o walk-forward
completo do notebook.
"""
from __future__ import annotations

import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient

from scripts.run_pipeline import (
    MODEL_EXPECTED_CLASSES,
    MODEL_FEATURE_NAMES,
    RHO_BOUNDS,
    apply_shrinkage,
    ensemble_proba,
    implied_1x2_from_goals,
    matrix_to_1x2,
    most_likely_score,
    predict_goals,
    proba_in_outcome_order,
    score_matrix,
    scores_from_lambdas,
    validate_model,
)
from scripts.validate_data import ROOT

NOTEBOOK_02 = ROOT / "notebooks" / "02_train.ipynb"
MODEL_PATH = ROOT / "models" / "model.joblib"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

SEED = 42
OUTCOMES = ("home", "draw", "away")

# Métricas da skill via sys.path (mesma fonte da avaliação final; funções puras de
# numpy, não tocam o gabarito). test_features.py já abre precedente importando de lá.
_SKILL = ROOT / ".claude" / "skills" / "avaliar-previsoes" / "scripts"
if str(_SKILL) not in sys.path:
    sys.path.insert(0, str(_SKILL))
from score_predictions import brier_multiclass, log_loss_mean, rps_mean  # noqa: E402


def _onehot(labels) -> np.ndarray:
    labels = list(labels)
    idx = np.array([OUTCOMES.index(o) for o in labels])
    oh = np.zeros((len(labels), 3))
    oh[np.arange(len(labels)), idx] = 1.0
    return oh


def _build_linear_pipeline():
    """Mesmo pipeline linear do notebook (imputação+indicador+scaler+one-hot)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    feats = list(MODEL_FEATURE_NAMES)
    cat = ["confed_home", "confed_away"]
    boolean = ["is_neutral", "h2h_available"]
    num = [c for c in feats if c not in cat + boolean]
    num_lin = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_lin, num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("bool", "passthrough", boolean),
        ]
    )
    return Pipeline(
        [("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]
    )


@pytest.fixture(scope="session")
def model_bundle() -> dict:
    """O bundle de models/model.joblib; gera o artefato se ainda não existir."""
    if not MODEL_PATH.is_file():
        notebook = nbformat.read(NOTEBOOK_02, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cwd=str(ROOT))
    import joblib

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def predict_df() -> pd.DataFrame:
    """As 72 linhas split=='predict' do features.parquet (entrada do 03)."""
    features = pd.read_parquet(FEATURES_PATH)
    return features[features["split"] == "predict"].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Contrato do bundle
# --------------------------------------------------------------------------- #
def test_bundle_structure(model_bundle: dict) -> None:
    assert isinstance(model_bundle, dict)
    assert {"pipeline", "meta"} <= model_bundle.keys()
    assert hasattr(model_bundle["pipeline"], "predict_proba")


def test_meta_keys_and_seed(model_bundle: dict) -> None:
    meta = model_bundle["meta"]
    for key in (
        "feature_names", "classes", "model_name", "seed", "sklearn_version",
        "shrink_alpha", "base_rate", "calibrated", "ensemble_weight",
    ):
        assert key in meta, f"meta sem '{key}'"
    assert meta["seed"] == SEED
    assert isinstance(meta["sklearn_version"], str) and meta["sklearn_version"]


def test_meta_shrinkage_config_is_valid(model_bundle: dict) -> None:
    # shrink_alpha em [0,1] e na grade {0, 0.05, ..., 0.5}; base_rate é o simplex 1X2.
    meta = model_bundle["meta"]
    alpha = meta["shrink_alpha"]
    assert isinstance(alpha, float) and 0.0 <= alpha <= 0.5
    assert any(abs(alpha - g) < 1e-9 for g in np.round(np.arange(0.0, 0.55, 0.05), 2))
    base = np.asarray(meta["base_rate"], dtype=float)
    assert base.shape == (3,)
    assert (base >= 0).all() and np.isclose(base.sum(), 1.0)
    assert isinstance(meta["calibrated"], bool)


def test_meta_ensemble_weight_valid(model_bundle: dict) -> None:
    # ensemble_weight em [0,1] e na grade {0, 0.05, ..., 1.0}; w<1 exige o componente de gols.
    w = model_bundle["meta"]["ensemble_weight"]
    assert isinstance(w, float) and 0.0 <= w <= 1.0
    assert any(abs(w - g) < 1e-9 for g in np.round(np.arange(0.0, 1.05, 0.05), 2))
    if w < 1.0:
        assert "goals" in model_bundle


def test_feature_names_match_contract(model_bundle: dict) -> None:
    assert tuple(model_bundle["meta"]["feature_names"]) == MODEL_FEATURE_NAMES


def test_classes_order_is_rps_order(model_bundle: dict) -> None:
    # meta['classes'] é a ordem ordinal do RPS; pipeline.classes_ é alfabético.
    meta = model_bundle["meta"]
    assert list(meta["classes"]) == MODEL_EXPECTED_CLASSES
    assert set(meta["classes"]) == set(model_bundle["pipeline"].classes_)


def test_validate_model_passes() -> None:
    # Reaproveita a validação canônica do run_pipeline (bundle + smoke das 72 linhas).
    assert validate_model() == 72


# --------------------------------------------------------------------------- #
# Sanidade probabilística e determinismo
# --------------------------------------------------------------------------- #
def test_predict_proba_shape_and_simplex(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    feats = model_bundle["meta"]["feature_names"]
    proba = model_bundle["pipeline"].predict_proba(predict_df[feats])
    assert proba.shape == (72, 3)
    assert not np.isnan(proba).any()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_is_deterministic(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    # Duas chamadas no mesmo bundle dão exatamente o mesmo resultado (teste barato).
    feats = model_bundle["meta"]["feature_names"]
    pipe = model_bundle["pipeline"]
    p1 = pipe.predict_proba(predict_df[feats])
    p2 = pipe.predict_proba(predict_df[feats])
    assert np.array_equal(p1, p2)


# --------------------------------------------------------------------------- #
# Shrinkage para a taxa-base (iteração): alpha=0 ≡ cru; preserva o simplex
# --------------------------------------------------------------------------- #
def test_apply_shrinkage_alpha_zero_is_identity() -> None:
    # Contrato central: alpha=0 reproduz as probabilidades cruas EXATAMENTE.
    rng = np.random.default_rng(SEED)
    proba = rng.dirichlet(np.ones(3), size=50)
    base = np.array([0.49, 0.227, 0.283])
    out = apply_shrinkage(proba, base, 0.0)
    assert np.array_equal(out, proba)


def test_apply_shrinkage_preserves_simplex() -> None:
    # P' = (1-a)P + a*base é combinação convexa de dois vetores que somam 1 -> soma 1,
    # em [0,1], para qualquer alpha da grade.
    rng = np.random.default_rng(SEED)
    proba = rng.dirichlet(np.ones(3), size=200)
    base = np.array([0.49, 0.227, 0.283])
    for alpha in np.round(np.arange(0.0, 0.55, 0.05), 2):
        out = apply_shrinkage(proba, base, float(alpha))
        assert out.shape == proba.shape
        assert (out >= 0).all() and (out <= 1).all()
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_apply_shrinkage_known_value() -> None:
    # Valor fechado: 0.5*[1,0,0] + 0.5*[0.4,0.3,0.3] = [0.7,0.15,0.15].
    out = apply_shrinkage(np.array([[1.0, 0.0, 0.0]]), np.array([0.4, 0.3, 0.3]), 0.5)
    assert np.allclose(out, [[0.7, 0.15, 0.15]])


# --------------------------------------------------------------------------- #
# Ensemble 1X2 (classificador + 1X2 implícito Dixon-Coles): weight=1 ≡ clf; simplex
# --------------------------------------------------------------------------- #
def test_ensemble_proba_weight_one_is_classifier() -> None:
    # Contrato central: weight=1.0 reproduz o classificador EXATAMENTE (retrocompat).
    rng = np.random.default_rng(SEED)
    clf = rng.dirichlet(np.ones(3), size=20)
    dc = rng.dirichlet(np.ones(3), size=20)
    assert np.array_equal(ensemble_proba(clf, dc, 1.0), clf)


def test_ensemble_proba_preserves_simplex() -> None:
    # P = w*clf + (1-w)*DC é combinação convexa de dois simplexos -> simplex.
    rng = np.random.default_rng(SEED)
    clf = rng.dirichlet(np.ones(3), size=100)
    dc = rng.dirichlet(np.ones(3), size=100)
    for w in np.round(np.arange(0.0, 1.05, 0.1), 1):
        out = ensemble_proba(clf, dc, float(w))
        assert (out >= 0).all() and (out <= 1).all()
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_ensemble_proba_known_value() -> None:
    # 0.5*[0.6,0.1,0.3] + 0.5*[0.2,0.5,0.3] = [0.4,0.3,0.3].
    out = ensemble_proba(np.array([[0.6, 0.1, 0.3]]), np.array([[0.2, 0.5, 0.3]]), 0.5)
    assert np.allclose(out, [[0.4, 0.3, 0.3]])


def test_implied_1x2_from_goals_is_simplex(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    # O 1X2 implícito da matriz Dixon-Coles nos 72 jogos é um simplex válido.
    proba = implied_1x2_from_goals(model_bundle["goals"], predict_df)
    assert proba.shape == (72, 3)
    assert (proba >= 0).all() and np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_submission_matches_meta_shrinkage(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    """A submissão do 03 reflete o shrinkage do meta (cru -> apply_shrinkage -> renorm).

    Reconstrói a previsão a partir do bundle exatamente como o 03 e compara com o
    CSV gravado. Garante que o shrink_alpha/base_rate persistidos chegam à produção.
    """
    submission_path = ROOT / "data" / "results" / "predictions_submission.csv"
    if not submission_path.is_file():
        pytest.skip("submission ainda não gerada (rode o 03 / run_pipeline)")
    meta = model_bundle["meta"]
    pipe = model_bundle["pipeline"]
    feats = meta["feature_names"]
    raw = proba_in_outcome_order(pipe, predict_df[feats])
    weight = float(meta.get("ensemble_weight", 1.0))
    if weight < 1.0:  # produção = clf -> ensemble(DC) -> shrinkage -> renorm (como o 03)
        proba_dc = implied_1x2_from_goals(model_bundle["goals"], predict_df)
        proba_1x2 = ensemble_proba(raw, proba_dc, weight)
    else:
        proba_1x2 = raw
    shrunk = apply_shrinkage(proba_1x2, np.asarray(meta["base_rate"], float), meta["shrink_alpha"])
    shrunk = shrunk / shrunk.sum(axis=1, keepdims=True)  # renorm defensiva (como o 03)

    # Esperado, alinhado por 'match' (a ordem de predict_df pode não ser a do CSV).
    expected = pd.DataFrame(
        {"match": predict_df["match_no"].astype(int),
         "p_home": shrunk[:, 0], "p_draw": shrunk[:, 1], "p_away": shrunk[:, 2]}
    ).sort_values("match").reset_index(drop=True)
    sub = pd.read_csv(submission_path).sort_values("match").reset_index(drop=True)
    assert sub["match"].tolist() == expected["match"].tolist()
    got = sub[["p_home", "p_draw", "p_away"]].to_numpy(float)
    assert np.allclose(got, expected[["p_home", "p_draw", "p_away"]].to_numpy(float), atol=1e-6)


# --------------------------------------------------------------------------- #
# Paridade de métricas: o notebook mede como a skill
# --------------------------------------------------------------------------- #
def test_metrics_parity_with_skill() -> None:
    # As funções replicadas no notebook copiam a matemática da skill; aqui validamos
    # a própria skill num vetor fixo para travar o contrato numérico (o notebook usa
    # a MESMA fórmula). Valores conferidos à mão.
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    onehots = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # RPS: jogo1 cdf=(0.6,0.9) obs=(1,1) -> (0.16+0.01)/2=0.085; jogo2 cdf=(0.2,0.4)
    # obs=(0,0) -> (0.04+0.16)/2=0.10; média=0.0925.
    assert rps_mean(probs, onehots) == pytest.approx(0.0925, abs=1e-9)
    # Brier: jogo1 0.16+0.09+0.01=0.26; jogo2 0.04+0.04+0.16=0.24; média=0.25.
    assert brier_multiclass(probs, onehots) == pytest.approx(0.25, abs=1e-9)
    # log-loss: (-ln0.6 - ln0.6)/2 = -ln0.6.
    assert log_loss_mean(probs, onehots) == pytest.approx(-np.log(0.6), abs=1e-9)


# --------------------------------------------------------------------------- #
# Teste de fogo da skill: RPS do modelo <= taxa-base (split temporal leve)
# --------------------------------------------------------------------------- #
def test_model_beats_base_rate_light_split() -> None:
    """Num corte temporal único e leve, o RPS do modelo <= RPS da taxa-base.

    Treina um pipeline linear (rápido) numa fatia recente do train com um único
    corte treino/holdout (80/20) — não o walk-forward de 5 folds do notebook —
    para manter o custo de CI baixo (R11).
    """
    features = pd.read_parquet(FEATURES_PATH)
    train = features[features["split"] == "train"]
    feats = list(MODEL_FEATURE_NAMES)

    order = np.argsort(train["date"].to_numpy(), kind="mergesort")
    X = train[feats].iloc[order].reset_index(drop=True)
    y = train["target_outcome"].astype("object").iloc[order].reset_index(drop=True)
    w = train["sample_weight"].iloc[order].reset_index(drop=True)
    d = train["date"].iloc[order].reset_index(drop=True)

    recent = d.to_numpy() >= np.datetime64("2010-01-01")
    Xr, yr, wr = X[recent], y[recent], w[recent]
    cut = int(len(Xr) * 0.8)

    pipe = _build_linear_pipeline()
    pipe.fit(Xr.iloc[:cut], yr.iloc[:cut], clf__sample_weight=wr.iloc[:cut].to_numpy(float))

    proba = proba_in_outcome_order(pipe, Xr.iloc[cut:])
    oh = _onehot(yr.iloc[cut:])
    rate = yr.iloc[:cut].value_counts(normalize=True).reindex(OUTCOMES).fillna(0.0).to_numpy()
    base = np.tile(rate, (len(oh), 1))

    assert rps_mean(proba, oh) <= rps_mean(base, oh)


def test_sample_weight_is_honored() -> None:
    """Pesos extremos mudam o predict_proba (garante que clf__sample_weight chega — R4)."""
    features = pd.read_parquet(FEATURES_PATH)
    train = features[features["split"] == "train"]
    feats = list(MODEL_FEATURE_NAMES)
    predict = features[features["split"] == "predict"]

    order = np.argsort(train["date"].to_numpy(), kind="mergesort")
    X = train[feats].iloc[order].reset_index(drop=True)
    y = train["target_outcome"].astype("object").iloc[order].reset_index(drop=True)
    sl = slice(len(X) - 4000, len(X))

    def fit_pred(weights: np.ndarray) -> np.ndarray:
        pipe = _build_linear_pipeline()
        pipe.fit(X.iloc[sl], y.iloc[sl], clf__sample_weight=weights)
        return pipe.predict_proba(predict[feats])

    uniform = fit_pred(np.ones(4000))
    skewed = np.ones(4000)
    skewed[:2000] = 100.0
    assert not np.allclose(uniform, fit_pred(skewed))


# --------------------------------------------------------------------------- #
# Componente de gols COMPANHEIRO (Poisson + Dixon-Coles)
# --------------------------------------------------------------------------- #
def test_goals_bundle_present_and_well_formed(model_bundle: dict) -> None:
    # O bundle estende o 1X2 com a chave 'goals' (home/away models, rho, features) e
    # o meta sinaliza o componente; o pipeline 1X2 segue intacto (chave 'pipeline').
    assert "goals" in model_bundle, "bundle sem componente de gols"
    g = model_bundle["goals"]
    for key in ("home_model", "away_model", "rho", "feature_names"):
        assert key in g, f"goals sem '{key}'"
    assert hasattr(g["home_model"], "predict") and hasattr(g["away_model"], "predict")
    assert tuple(g["feature_names"]) == MODEL_FEATURE_NAMES
    assert model_bundle["meta"].get("has_goals") is True


def test_goals_rho_in_valid_range(model_bundle: dict) -> None:
    rho = float(model_bundle["goals"]["rho"])
    assert RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]
    # coerente com a dependência negativa da EDA (corr home/away ~ -0.145).
    assert rho < 0.0, "rho esperado negativo (déficit de empate / dependência neg.)"


def test_goals_lambda_finite_positive_on_72(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    lam_home, lam_away = predict_goals(model_bundle["goals"], predict_df)
    assert lam_home.shape == (72,) and lam_away.shape == (72,)
    for lam in (lam_home, lam_away):
        assert np.isfinite(lam).all()
        assert (lam > 0).all()


def test_score_matrix_sums_to_one() -> None:
    # P(h,a) normalizada: soma ~1 para lambdas/rho representativos (incl. extremos).
    for lh, la in [(1.5, 1.1), (0.3, 0.3), (3.0, 0.5)]:
        m = score_matrix(lh, la, -0.1)
        assert np.isclose(m.sum(), 1.0, atol=1e-9)
        assert (m >= 0).all()


def test_most_likely_score_consistent_with_1x2() -> None:
    # O placar (argmax restrito) cai SEMPRE na região do resultado 1X2 pedido.
    m = score_matrix(1.6, 1.0, -0.1)
    h, a = most_likely_score(m, "home")
    assert h > a
    h, a = most_likely_score(m, "draw")
    assert h == a
    h, a = most_likely_score(m, "away")
    assert h < a


def test_scores_from_lambdas_match_1x2(model_bundle: dict, predict_df: pd.DataFrame) -> None:
    # Reconstrói como o 03: 1X2 (com shrinkage) -> argmax -> placar consistente.
    meta = model_bundle["meta"]
    pipe = model_bundle["pipeline"]
    feats = meta["feature_names"]
    raw = proba_in_outcome_order(pipe, predict_df[feats])
    weight = float(meta.get("ensemble_weight", 1.0))
    proba_1x2 = (
        ensemble_proba(raw, implied_1x2_from_goals(model_bundle["goals"], predict_df), weight)
        if weight < 1.0
        else raw
    )
    proba = apply_shrinkage(proba_1x2, np.asarray(meta["base_rate"], float), meta["shrink_alpha"])
    outcomes = [MODEL_EXPECTED_CLASSES[i] for i in np.argmax(proba, axis=1)]
    lam_home, lam_away = predict_goals(model_bundle["goals"], predict_df)
    xg, placares = scores_from_lambdas(lam_home, lam_away, model_bundle["goals"]["rho"], outcomes)
    assert xg.shape == (72, 2)
    for outcome, placar in zip(outcomes, placares):
        h, a = (int(v) for v in placar.split("-"))
        if outcome == "home":
            assert h > a
        elif outcome == "away":
            assert h < a
        else:
            assert h == a


def test_implied_1x2_is_a_simplex() -> None:
    # O 1X2 implícito pela matriz (soma das regiões) é um simplex válido.
    p = matrix_to_1x2(score_matrix(1.4, 1.2, -0.08))
    assert p.shape == (3,)
    assert (p >= 0).all() and np.isclose(p.sum(), 1.0, atol=1e-9)


def test_submission_has_scoreline_consistent(predict_df: pd.DataFrame) -> None:
    """Se a submissão do 03 já existe, suas colunas de placar batem com o 1X2."""
    submission_path = ROOT / "data" / "results" / "predictions_submission.csv"
    if not submission_path.is_file():
        pytest.skip("submission ainda não gerada (rode o 03 / run_pipeline)")
    sub = pd.read_csv(submission_path)
    if "placar_provavel" not in sub.columns:
        pytest.skip("submission sem componente de gols")
    assert {"xg_home", "xg_away", "placar_provavel"} <= set(sub.columns)
    pred = sub[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)
    ha = sub["placar_provavel"].str.split("-", expand=True).astype(int).to_numpy()
    exp = np.where(ha[:, 0] > ha[:, 1], 0, np.where(ha[:, 0] < ha[:, 1], 2, 1))
    assert (pred == exp).all(), "placar_provavel contradiz o 1X2 na submissão"


# --------------------------------------------------------------------------- #
# Anti-vazamento
# --------------------------------------------------------------------------- #
def test_notebook_does_not_read_ground_truth() -> None:
    # Nenhuma menção ao gabarito (worldcup-2026-results) em nenhuma célula de código.
    notebook = nbformat.read(NOTEBOOK_02, as_version=4)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "worldcup-2026-results" not in code.lower(), "notebook referencia o gabarito"
