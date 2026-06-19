# Tarefas — Treino (`02_train.ipynb`)

Checklist acionável do plano em [`README.md`](README.md), por etapas E0–E7
(= seções §6.0–§6.7). **Otimizar para o RPS; sempre comparar com baselines;
nunca usar `worldcup-2026-results.csv`.**

## E0 — Setup, carga e separação (§6.0)
- [ ] Reaproveitar setup do esqueleto (`find_root`, caminhos, `MODELS.mkdir`).
- [ ] Ler `features.parquet`; separar `train` (split=="train"); **não tocar** em `predict`.
- [ ] Montar `y = target_outcome`, `w = sample_weight`, `dates = date`.
- [ ] Congelar `FEATURE_NAMES` (as 22 colunas da §8) e montar `X = train[FEATURE_NAMES]`.

## E1 — Harness de avaliação (§6.1, §10.4) — *antes de treinar*
- [ ] Replicar `rps_mean`, `brier_multiclass`, `log_loss_mean`, `accuracy` da skill (mesma ordem `OUTCOMES=("home","draw","away")`) **+ teste de paridade** contra `score_predictions` num vetor fixo (§10.6).
- [ ] **Ordenar o `train` por data uma vez** (`np.argsort(..., kind="mergesort")`), fora do loop de folds (R6).
- [ ] Implementar split **walk-forward** (`TimeSeriesSplit` sobre `train` ordenado) — **nunca** K-fold aleatório.
- [ ] Função `evaluate(estimator, X, y, w, dates, n_splits=5)` (§10.4) com `clone(estimator)` por fold; `sample_weight` via **`clf__sample_weight`** (verificar que chega ao estimador — R4); pré-proc só no fold; retorna média entre folds + detalhe por fold.
- [ ] `predict_proba` reordenado **por nome** (`classes_` alfabético → `OUTCOMES`) por uma **função única** reusada em harness/03/validate (R1).
- [ ] Avaliação **sem peso** (imita os 72 jogos); peso só no fit (R5).
- [ ] Baselines no mesmo split: **taxa-base** (estimada só com `y` do treino do fold), **sempre-mandante**, **maior-ranking**.
- [ ] Reportar também a fatia **≥1993** (mesma `evaluate`, recorte `date>=1993`); **priorizar essa fatia na seleção final** (nota de engenharia §11).

## E2 — Definição do alvo (§6.2)
- [ ] Fixar alvo = **classificador 1X2** (`home/draw/away`); registrar a decisão e a alternativa DC.

## E3 — Pré-processamento (§6.3, §10.1–§10.3)
- [ ] Derivar `NUM/CAT/BOOL_FEATURES` de `FEATURE_NAMES` por **diferença de conjuntos** (não redigitar) + `assert` de cobertura (§10.1).
- [ ] **Não** fazer `.astype()` fora do `Pipeline`: sklearn consome `boolean`/`string` nullable do parquet cru (verificado — R9). Conversão p/ `category` (HistGB nativo), se usada, **dentro** do `Pipeline`.
- [ ] `ColumnTransformer` linear: numéricas (`SimpleImputer(median, add_indicator=True)` + `StandardScaler`), categóricas (`OneHotEncoder(handle_unknown="ignore")` em `confed_*`), booleanas `passthrough`.
- [ ] Variante HistGB: **sem** imputação/scaler (NaN cru); decidir entre one-hot e `categorical_features="from_dtype"` (§10.3).
- [ ] Garantir que o pré-proc vive **dentro** do `Pipeline` (ajuste só no treino do fold — R6).

## E4 — Baseline forte (§6.4)
- [ ] `LogisticRegression(max_iter=1000, random_state=SEED)` em `Pipeline` (imputação + scaler + one-hot). **Sem** `multi_class` (removido na 1.7 — R2); multinomial é o default.
- [ ] Avaliar com o harness; comparar com os 3 baselines; testar `class_weight="balanced"` (manter só se melhora RPS).

## E5 — Alternativas (§6.5)
- [ ] `HistGradientBoostingClassifier` (aceita NaN nativo; `sample_weight` no fit).
- [ ] (Opcional) modelo de gols Poisson/Dixon-Coles → derivar 1X2; só se houver ganho de RPS medido.
- [ ] Montar a tabela `modelo × {RPS, Brier, log-loss, acc}` vs. baselines.

## E6 — Calibração, seleção e persistência (§6.6, §10.5, §10.7)
- [ ] Calibrar o melhor candidato (`CalibratedClassifierCV`, `cv=TimeSeriesSplit(...)` — **nunca** `cv=int` nem `cv="prefit"`; isotônica se houver dados, senão sigmoide).
- [ ] **Medir RPS antes/depois** e decidir por **delta** (leak parcial da calibração torna o absoluto otimista — R3); manter só se melhora.
- [ ] Selecionar pelo **menor RPS de validação** (priorizar fatia ≥1993; desempate: calibração, depois parcimônia).
- [ ] (Se testado) registrar resultado do **aumento simétrico** — adotar só com ganho medido.
- [ ] Re-treinar o pipeline escolhido em **todo** o `train` (com `sample_weight`).
- [ ] Montar `meta` (`feature_names`, `classes=["home","draw","away"]`, `model_name`, `seed=42`, `sklearn_version`, `created_at` via `datetime.now(timezone.utc)`, `cv_rps`).
- [ ] `joblib.dump({"pipeline":..., "meta":...}, MODELS / "model.joblib")`.
- [ ] `mkdir docs/train/figuras/` (ainda não existe) e salvar tabela de resultados + figura de calibração lá.

## E7 — Validação do artefato e testes (§6.7, §10.8, §10.9)
- [ ] `validate_model()` no `run_pipeline.py` + registrar em `ARTIFACT_VALIDATORS["02_train.ipynb"]` (espelhar `validate_features`: levanta `ValidationError`, retorna `int`).
- [ ] Adicionar `import joblib` e `import numpy as np` ao topo do `run_pipeline.py` (hoje só importa pandas/nbformat/nbclient) e **declarar `joblib` no `requirements.txt`** (não confiar na transitividade do sklearn).
- [ ] `validate_model` checa: desserializa; dict com `pipeline`+`meta`; `meta["classes"]==["home","draw","away"]` e `set(...)==set(pipeline.classes_)`; `feature_names` ⊆ colunas do parquet; **smoke** `predict_proba` nos 72 → `(72,3)`, sem NaN, em `[0,1]`, soma ~1 (R7/R8).
- [ ] `tests/test_train.py` (fixture de sessão que gera o artefato se ausente): contrato do bundle, `seed==42`, `sklearn_version` presente, ordem das classes, sanidade probabilística, **RPS ≤ taxa-base** num split leve, determinismo (duas chamadas de `predict_proba` idênticas), grep do gabarito, paridade de métricas (§10.6).
- [ ] (Opcional) teste de que `sample_weight` extremo muda a saída (garante que o peso é honrado — R4).
- [ ] `ruff check scripts tests` + `pytest` + `run_pipeline` verdes (CI roda os três; pipeline **sem warnings** — R2).

## Ajuste no `03` (consumidor do artefato)
- [ ] Carregar bundle; selecionar `predict[meta["feature_names"]]`; `predict_proba`.
- [ ] **Mapear colunas por nome** (`meta["classes"]`/`pipeline.classes_`) → `p_home/p_draw/p_away` (nunca por posição).
- [ ] Renormalizar (defensivo) e escrever `predictions_submission.csv` no contrato da skill (`match, home, away, p_home, p_draw, p_away`); `match` ← `match_no`.

## Conferência final
- [ ] Notebook reexecuta de cima a baixo sem erro (`Restart & Run All`).
- [ ] Modelo final **vence a taxa-base em RPS** (Definition of Done).
- [ ] Nenhuma célula lê `worldcup-2026-results.csv`.
- [ ] `model.joblib` desserializa e alimenta o `03` sem ajuste extra.
