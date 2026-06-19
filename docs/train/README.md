# Plano de Execução — Treino (`notebooks/02_train.ipynb`)

Plano da etapa de treino do Paul the Octopus. O `02` lê o `features.parquet` do
`01`, treina e seleciona um modelo de previsão **1X2** (vitória do mandante /
empate / vitória do visitante) e grava `models/model.joblib`. O `03` apenas
carrega esse artefato e aplica `predict_proba` às 72 linhas `split=="predict"`.

Pré-requisito de leitura:

- A **EDA** ([`docs/eda/README.md`](../eda/README.md)) — taxas-base, distribuição
  de gols, mando × campo neutro, ruído de amistosos, ranking só ≥1993.
- As **features** ([`docs/features/README.md`](../features/README.md) e
  [`docs/features/decisoes.md`](../features/decisoes.md)) — o contrato exato do
  `features.parquet` que este notebook consome.
- A **skill de avaliação**
  ([`.claude/skills/avaliar-previsoes/SKILL.md`](../../.claude/skills/avaliar-previsoes/SKILL.md)
  e [`references/metricas.md`](../../.claude/skills/avaliar-previsoes/references/metricas.md))
  — o contrato da submissão, as métricas (RPS principal) e os baselines. **O `02`
  é otimizado para a métrica que a skill mede.**

> **Decisão central deste plano:** o `model.joblib` é um **`Pipeline` do
> scikit-learn completo e autossuficiente** (pré-processamento + estimador
> calibrado), treinado nas linhas `split=="train"`, cuja saída é
> `predict_proba → (p_home, p_draw, p_away)` na **ordem ordinal do RPS**
> (`home → draw → away`). Toda a ciência de modelagem vive aqui; o `03` não
> conhece feature, scaler nem encoder — só dá `joblib.load(...).predict_proba(X)`.

> **Por que o `02` decide tudo isto agora:** a `decisoes.md` do `01`
> deliberadamente empurrou para cá escalonamento/one-hot, imputação de NaN,
> aumento simétrico, **escolha do algoritmo** (classificador 1X2 vs. modelo de
> gols), tuning e eventual filtro de janela de treino. Este é o documento que as
> resolve — com recomendação, prós/contras e medição.

---

## 1. Objetivos

1. Produzir `models/model.joblib` — um `Pipeline` sklearn **fechado**
   (pré-processamento + estimador calibrado) que mapeia as colunas de
   `features.parquet` para probabilidades 1X2 calibradas.
2. **Otimizar para o RPS** (métrica principal da skill), reportando também
   Brier, log-loss, acurácia e calibração — nunca só acurácia.
3. **Bater os três baselines** da skill (taxa-base histórica, sempre-mandante,
   maior-ranking). Um modelo que não vence a taxa-base não agregou sinal.
4. **Validação sem vazamento temporal** (walk-forward), com pré-processamento
   ajustado só no treino de cada fold e `sample_weight` honrado no fit.
5. Persistir **metadados** suficientes para o `03` ser trivial e auditável
   (ordem das features, ordem das classes, versão, semente).
6. Acrescentar a validação do `model.joblib` ao `run_pipeline.py` (E7-equivalente)
   e testes em `tests/` — o CI roda `ruff` + `pytest`.

## 2. Escopo, entrada e contrato de saída

**Entrada:** `data/processed/features.parquet` (49.400 linhas `train` + 72
`predict`; schema na [`decisoes.md §10`](../features/decisoes.md)). **O `02` só
treina nas linhas `train`.** As 72 `predict` não são tocadas aqui — são do `03`.

**Nunca** ler `data/raw/worldcup-2026-results.csv` (gabarito). A avaliação contra
resultados reais é da skill `avaliar-previsoes`, **fora** deste notebook.

### 2.1 Contrato de `models/model.joblib`

Um único objeto persistido com `joblib.dump`. Recomendação: **um dicionário**
`{"pipeline": ..., "meta": ...}` (não o estimador cru), para carregar metadados
sem inferir nada do objeto sklearn. O `03` faz:

```python
bundle  = joblib.load(MODELS / "model.joblib")
pipe    = bundle["pipeline"]
feats   = bundle["meta"]["feature_names"]      # ordem exata de entrada
classes = bundle["meta"]["classes"]            # ("home","draw","away")
proba   = pipe.predict_proba(predict[feats])   # colunas na ordem de classes
```

| Chave | Conteúdo | Por quê |
|---|---|---|
| `pipeline` | `Pipeline`/`CalibratedClassifierCV` já **fitted** no `train` inteiro | `03` só dá `predict_proba`; sem train/serve skew |
| `meta["feature_names"]` | lista ordenada das colunas de entrada (§8) | `03` seleciona `X` na ordem certa; falha cedo se faltar coluna |
| `meta["classes"]` | `["home","draw","away"]` (**ordem do RPS**) | mapear colunas de `predict_proba` → `p_home/p_draw/p_away` sem ambiguidade |
| `meta["model_name"]` | rótulo do modelo escolhido (ex.: `"logreg_multinomial_isotonic"`) | rastreabilidade |
| `meta["seed"]` | `42` | reprodutibilidade |
| `meta["sklearn_version"]`, `meta["created_at"]` | versão e timestamp | guardrail de compatibilidade ao desserializar |
| `meta["cv_rps"]` (opcional) | RPS médio na validação walk-forward | sanity check; o `03`/relatório lê sem re-treinar |

**Guardrail da ordem das classes (crítico).** `predict_proba` ordena colunas por
`pipe.classes_`, que vem de `np.unique` (alfabético: `away, draw, home`). Isso
**não** é a ordem do RPS. O `03` (e a validação do artefato) deve **mapear por
nome** via `meta["classes"]`/`pipe.classes_`, nunca assumir posição. Persistir
`classes` em `meta` torna esse mapeamento explícito e testável.

> **Ajuste no `03` (já previsto no esqueleto):** carregar o bundle, selecionar
> `predict[feature_names]`, dar `predict_proba`, mapear colunas por nome para
> `p_home/p_draw/p_away`, **renormalizar** (defensivo) e escrever
> `predictions_submission.csv` com as colunas do contrato da skill
> (`match, home, away, p_home, p_draw, p_away`). `match` vem de `match_no`.

## 3. Princípios e guardrails

O risco número 1 desta etapa é **medir errado** (vazamento ou métrica enganosa) e
concluir que o modelo é bom quando não é. Regras invioláveis:

- **Anti-vazamento temporal.** Validação respeita a ordem do tempo (walk-forward,
  §5) — **nunca** K-fold aleatório. Todo pré-processamento (imputação, scaler,
  encoder) é ajustado **só no treino de cada fold**, dentro do `Pipeline` —
  nunca no conjunto inteiro antes de dividir.
- **Otimizar a métrica certa.** Seleção de modelo e tuning são decididos pelo
  **RPS** de validação (a métrica da skill), não pela acurácia.
- **Sempre comparar com baseline.** Todo candidato é reportado lado a lado com
  taxa-base / sempre-mandante / maior-ranking. Sem isso, o número não significa
  nada.
- **Calibração é requisito, não enfeite.** A saída são probabilidades que vão
  para Brier/log-loss/RPS; probabilidades descalibradas perdem nessas métricas.
- **Reprodutibilidade.** `SEED = 42` em tudo que sorteia (estimador, calibração,
  splits determinísticos); `find_root()` já no esqueleto; sem estado global.
- **Agnóstico ao `03`.** Tudo que o `03` precisa para reproduzir a predição mora
  no bundle (pipeline + meta). O `03` não recomputa nada.
- **Nunca o gabarito.** `worldcup-2026-results.csv` não é lido aqui (grep no E7).

### Fatos dos dados que guiam a modelagem (medidos no `features.parquet`)

| Fato medido | Implicação para o `02` |
|---|---|
| Alvo: home **49,0%**, draw **22,7%**, away **28,3%** | desbalanceado; **empate é a classe minoritária e a mais difícil** — exatamente onde o RPS cobra. Não usar acurácia como guia. |
| **41,5%** do `train` tem `rank_*`/`points_diff` **NaN** (pré-1993) | decisão de NaN é central: imputar (linear) **vs.** estimador que aceita NaN nativo (§7). |
| `form_*`/`rest_days_*`: ~142–195 NaN (estreias de seleção) | poucos; imputar trivial. |
| `predict` (72 jogos): **0 NaN** em Elo, ranking, forma, rest, confed | os 72 jogos **sempre** têm preditores; o NaN é problema só do `train` antigo. |
| Stack: sklearn **1.7.1**, **sem** XGBoost/LightGBM | boosting = `HistGradientBoostingClassifier` (nativo, **aceita NaN**); não adicionar dependência sem necessidade medida. |
| `confed_*`: 7 categorias (6 confederações + `"unknown"`), sem NaN | one-hot barato e seguro; `"unknown"` é um nível como outro qualquer. Verificado: `['AFC','CAF','CONCACAF','CONMEBOL','OFC','UEFA','unknown']`. |
| `is_neutral` em predict: **63 neutros / 9 não-neutros** | mando importa pouco em 2026; o modelo precisa **separar mando de força** (peso de `is_neutral`), não confundir. |
| `tournament_tier` em predict: **todos = 5** (Copa) | tier não diferencia os 72 jogos; entra pelo `train` (ajuda a aprender que jogo competitivo ≠ amistoso), não como discriminante de 2026. |
| ~62% do `train` é ≥1993; `sample_weight` de 0 a 2,42 (mediana 0,14) | o peso já desvaloriza amistoso/antigo; um filtro de janela é alternativa **medida**, não default. |

## 4. Plano geral (etapas)

Cada etapa é uma seção do notebook (§6) e tem tarefas em
[`tarefas.md`](tarefas.md). A ordem importa: **definir a avaliação antes de
treinar** evita escolher modelo pela métrica errada.

| Etapa | Seção | Entrega |
|---|---|---|
| **E0** Setup, carga e separação `train`/`predict` | §6.0 | `X`, `y`, `w` do `train`; lista de features congelada |
| **E1** Harness de avaliação (RPS + baselines + walk-forward) | §6.1 | função que pontua qualquer modelo do mesmo jeito que a skill |
| **E2** Definição do alvo | §6.2 | alvo = 1X2 (classificador) — decisão registrada |
| **E3** Pré-processamento (`ColumnTransformer`) | §6.3 | imputação/scaler/one-hot — ajuste só no fold |
| **E4** Baseline forte (regressão logística multinomial) | §6.4 | 1º modelo medido vs. baselines |
| **E5** Alternativas (boosting; opcional modelo de gols) | §6.5 | 1–2 candidatos comparados por RPS |
| **E6** Calibração + seleção + ajuste final + persistência | §6.6 | `model.joblib` (pipeline + meta) |
| **E7** Validação do artefato + testes | §6.7 | checagem no `run_pipeline` + `pytest` |

## 5. Tópicos

Os eixos transversais que o `02` precisa cobrir:

- **Métrica primeiro** (RPS como bússola; Brier/log-loss/acurácia/calibração de apoio).
- **Validação temporal honesta** (walk-forward; pré-processamento dentro do fold).
- **Baselines obrigatórios** (a barra a superar).
- **Tratamento de NaN e desbalanceamento** (41,5% sem ranking; empate minoritário).
- **Calibração de probabilidades** (a saída são probabilidades, não rótulos).
- **Simplicidade com ganho medido** (baseline forte antes de complexidade).
- **Reprodutibilidade e contrato de saída** (bundle autossuficiente para o `03`).

## 6. Seções e decisões (detalhado)

### §6.0 — Setup, carga e separação (E0)

Reaproveitar o setup do esqueleto (`find_root`, caminhos, `MODELS.mkdir`). Ler
`features.parquet`; separar `train = features[split=="train"]` e **não tocar** em
`predict`. Montar:

- `y = train["target_outcome"]` (rótulos `home/draw/away`).
- `w = train["sample_weight"]` (peso de treino).
- `X = train[FEATURE_NAMES]` — a lista **congelada** de preditores (§8), que vira
  `meta["feature_names"]`.
- `dates = train["date"]` — eixo do split temporal (E1); **não é feature**.

**Saída:** `X`, `y`, `w`, `dates` e a lista de features fixada.

### §6.1 — Harness de avaliação (E1) — *fazer antes de treinar*

Construir **uma** rotina que pontua qualquer modelo do mesmo jeito que a skill, e
que será usada em todos os candidatos (comparação justa):

1. **Métricas idênticas às da skill.** Reusar/replicar `rps_mean`,
   `brier_multiclass`, `log_loss_mean`, `accuracy` de
   `score_predictions.py` (mesma matemática, mesma ordem `OUTCOMES`). Não
   reinventar — assim o número do `02` casa com o número da avaliação final.
2. **Split temporal walk-forward** (ver §6 abaixo, item de validação). Para cada
   fold: treina no passado, avalia no futuro, com pré-processamento ajustado só no
   treino e `sample_weight` no `.fit(...)`.
3. **Baselines no mesmo split** (taxa-base, sempre-mandante, maior-ranking) — a
   taxa-base estimada **só com o treino de cada fold** (não com o histórico
   inteiro, senão vaza o futuro do fold).

**Saída:** `evaluate(model) -> {rps, brier, log_loss, acc}` médios por fold + a
mesma coisa para os baselines. É a régua única do notebook.

### §6.2 — Definição do alvo (E2)

Ver §7.A. **Recomendado: classificador 1X2 direto** (3 classes ordinais
`home/draw/away`). O `01` já entrega `target_outcome` pronto. A rota de modelo de
gols (Poisson/Dixon-Coles) fica registrada como alternativa medível (placares
estão no artefato), não como caminho padrão.

### §6.3 — Pré-processamento (E3)

Um `ColumnTransformer` (ver §7.C/§8), **dentro** do `Pipeline`, com três grupos:
numéricas (imputar + escalonar para modelos lineares), categóricas (one-hot de
`confed_*`), e booleanas (passthrough/inteiro). Ajustado só no treino de cada
fold. **Saída:** transformador reutilizável.

### §6.4 — Baseline forte: regressão logística multinomial (E4)

Primeiro modelo **real**: `LogisticRegression(max_iter=1000)` num `Pipeline` com
imputação + `StandardScaler` + one-hot. Interpretável, rápido, honesto com poucos
dados. Medir com o harness (§6.1) e comparar com os 3 baselines. **Saída:** 1ª
linha da tabela de resultados; a barra interna a bater.

> **Nota de engenharia (verificado na stack — sklearn 1.7.1):** **não** passar
> `multi_class="multinomial"`. Esse argumento foi **deprecado na 1.5 e removido na
> 1.7** (emite `FutureWarning` e some na sequência); a `LogisticRegression` já usa
> multinomial por padrão em problema multiclasse. Manter a assinatura sem
> `multi_class`. `max_iter` default (100) pode não convergir com 18+ features
> escalonadas — usar `max_iter=1000` para evitar `ConvergenceWarning` (o CI roda
> com warnings; um pipeline que cospe warning é ruído de diagnóstico).

### §6.5 — Alternativas (E5)

1–2 candidatos, comparados pelo **mesmo** harness:

- **`HistGradientBoostingClassifier`** (sklearn nativo). Captura não-linearidades e
  interações (ex.: `is_neutral` × `elo_diff`), **aceita NaN nativamente** (resolve
  os 41,5% sem ranking sem imputar) e não precisa de scaler. É o boosting "sem
  custo de dependência".
- **(Opcional, registrado) modelo de gols Poisson/Dixon-Coles** → derivar 1X2
  (§7.A). Só vale o esforço se houver ganho de RPS medido; senão, fica como nota.

**Saída:** tabela `modelo × {RPS, Brier, log-loss, acc}` vs. baselines.

### §6.6 — Calibração, seleção, ajuste final e persistência (E6)

1. **Calibração** (§7.F): envolver o melhor candidato em `CalibratedClassifierCV`
   (isotônica se houver dados; Platt/sigmoide se escassez) e **medir o RPS antes e
   depois** — só manter a calibração se melhorar.
2. **Seleção** (§7.E): escolher pelo **menor RPS de validação**, com Brier/calibração
   como desempate e parcimônia como critério final (modelo mais simples vence
   empate técnico).
3. **Ajuste final:** re-treinar o pipeline escolhido em **todo** o `train` (com
   `sample_weight`).
4. **Persistir** o bundle `{"pipeline", "meta"}` em `models/model.joblib` (§2.1).

**Saída:** `model.joblib`.

### §6.7 — Validação do artefato e testes (E7)

Acrescentar `validate_model()` ao `run_pipeline.py` e registrá-lo em
`ARTIFACT_VALIDATORS["02_train.ipynb"]`; adicionar testes em `tests/` (ver §9).
**Saída:** `run_pipeline` + `pytest` + `ruff` verdes.

## 7. Decisões em aberto — com recomendação

### 7.A — Definição do alvo: classificador 1X2 vs. modelo de gols

| Critério | **Classificador 1X2** (recomendado) | Modelo de gols (Poisson/Dixon-Coles) |
|---|---|---|
| Alinhamento à métrica | otimiza direto o que vira RPS (probas 1X2) | otimiza gols; 1X2 é derivado (objetivo indireto) |
| Empate | classe própria, peso via `class_weight`/`sample_weight` | empate emerge da diagonal; **DC corrige** o déficit de empate em placares baixos |
| EDA | — | favorável: gols ~Poisson, leve dependência negativa (empate obs. **0,2274** vs **0,2349** sob independência; corr. home/away ≈ **−0,145**) |
| Esforço/risco | baixo; `target_outcome` pronto; sklearn direto | alto; sem lib pronta na stack (implementar verossimilhança/otimização à mão) |
| Calibração de probas | direta (`predict_proba` + calibração) | indireta; precisa montar a matriz de placares e somar a diagonal |

**Recomendação:** **classificador 1X2 direto** como caminho principal — alinhado à
métrica, baixo risco, e o desbalanceamento de empate é tratável com peso. **Manter
Dixon-Coles como alternativa explicitamente medível** (não descartada): o achado da
EDA (déficit de empate + dependência negativa) é *exatamente* o que o termo de
correção de DC endereça, então é a hipótese mais promissora **se** o classificador
1X2 subestimar empates na calibração. Critério objetivo para investir em DC: se,
na fração de validação, o modelo 1X2 calibrado tiver RPS pior que a taxa-base **na
fatia de jogos equilibrados** (|elo_diff| pequeno, onde o empate é mais provável),
testar DC. Caso contrário, 1X2 basta. *(Confirmar a leitura com o
`football-analyst`: o empate em jogos truncados/equilibrados de Copa é o cenário
onde DC historicamente ajuda.)*

### 7.B — Algoritmos candidatos

| Modelo | Papel | Escala? | NaN? | Prós | Contras |
|---|---|---|---|---|---|
| **LogisticRegression** (multinomial por padrão; **sem** `multi_class`, removido na 1.7) | **baseline forte** | **sim** (StandardScaler) | **não** → imputar | interpretável, rápido, calibra bem, robusto com pouca amostra; coeficientes auditáveis | só linear (precisa de features de interação manuais p/ `is_neutral`×força) |
| **HistGradientBoostingClassifier** | alternativa principal | não | **sim, nativo** | não-linear + interações automáticas; **aceita NaN** (mata os 41,5%); suporta `sample_weight` | menos interpretável; risco de overfit com cauda antiga de baixo peso; calibração às vezes pior |
| **Poisson/Dixon-Coles** | alternativa registrada (7.A) | n/a | n/a | principiado p/ gols/empate; placares + probas coerentes | sem lib na stack; mais código e risco |

**Decisão de pré-processamento que decorre:** só os modelos **lineares** exigem
escalonamento e imputação; o HistGB dispensa ambos. Por isso o pré-processamento
**faz parte do `Pipeline` de cada candidato** (não é global) — cada modelo recebe
o pré-processamento que precisa, e a comparação continua justa porque o harness
(§6.1) avalia todos no mesmo split.

### 7.C — Pré-processamento (decisão concreta de `ColumnTransformer`)

- **Numéricas** (Elo, rank/points/diff, form, rest, h2h, tier):
  - *Pipeline linear:* `SimpleImputer(strategy="median")` **+** indicador de
    ausência (`add_indicator=True`, importante para o ranking pré-1993 — a
    ausência carrega sinal de época) **+** `StandardScaler`.
  - *Pipeline HistGB:* **sem imputação e sem scaler** (passa NaN cru; o modelo
    trata).
- **Categóricas** (`confed_home`, `confed_away`): `OneHotEncoder(handle_unknown=
  "ignore")`. 7 categorias (6 confederações + `"unknown"`), sem NaN → barato.
  `handle_unknown="ignore"` é defensivo (nível inédito em predict vira zeros), mas
  **verificado:** as confederações dos 72 jogos já aparecem no `train`, então não há
  categoria nova em produção. (`tournament` cru **não** entra; o `01` já o reduziu a
  `tournament_tier` ordinal.) *Alternativa para o HistGB:* em vez de one-hot,
  passar `confed_*` como `dtype="category"` e usar `categorical_features=
  "from_dtype"` — o HistGB trata categóricas nativamente (split por subconjunto),
  mais eficiente que o one-hot. Decidir no §6.5.
- **Booleanas** (`is_neutral`, `h2h_available`): converter para `int` (0/1);
  passthrough.
- **Fora de `X`:** `split`, `match_no`, `date`, `home`, `away`, `target_outcome`,
  `home_score`, `away_score`, `sample_weight` (chaves/alvo/peso — §8).

### 7.D — Validação SEM vazamento temporal

**Por que K-fold aleatório vaza:** sortear linhas mistura jogos de 2025 no treino
e de 2010 no teste — o modelo "vê o futuro" e a métrica fica otimista (é a causa
clássica da "acurácia de 94% suspeita"). Em séries temporais o teste tem de ser
**sempre posterior** ao treino.

**Recomendado: walk-forward por blocos temporais.** Ordenar por `date` e avaliar
em janelas crescentes — treina em `[início, t)`, valida em `[t, t+Δ)`, avança `t`.
Opções equivalentes e aceitáveis:

- `TimeSeriesSplit(n_splits=5)` sobre o `train` **já ordenado por data** (simples,
  pronto no sklearn). É o caminho default.
- *Folds por era de Copa* (ex.: validar em 2014, 2018, 2022, 2026-pré) — mais
  interpretável ("o modelo teria acertado a Copa X?"), recomendado como **corte de
  reporte** além do `TimeSeriesSplit`.

**Restrições do split:**

- **Pré-processamento ajustado só no treino do fold** (garantido por estar dentro
  do `Pipeline` passado ao split — o sklearn faz `fit` no treino e `transform` no
  teste automaticamente). Nunca `fit` no `X` inteiro antes de dividir.
- **Para a fração mais recente do `train`** (≥1993, com ranking) reportar uma
  métrica à parte: é a fatia mais parecida com 2026 e o número mais honesto para
  prever a Copa. O modelo final treina em tudo (com peso), mas a **leitura** de
  qualidade prioriza essa fatia.

**Papel do `sample_weight`:**

- **No fit:** passar `sample_weight=w` a `.fit(...)` (e propagar pelo `Pipeline`
  com `nome_do_passo__sample_weight`). Reflete a decisão do `01` (amistoso/antigo
  pesam menos).
- **Na avaliação:** **recomendado avaliar SEM peso** (RPS não-ponderado), porque a
  skill pontua os 72 jogos de 2026 sem peso — a métrica de validação deve imitar a
  de produção. Reportar a versão ponderada como secundária, se útil. *(Decisão a
  registrar explicitamente no notebook.)*

### 7.E — Métricas e critério de seleção

- **Principal: RPS** (a métrica da skill; respeita a ordem 1-X-2). Menor = melhor.
- **Apoio:** Brier e log-loss (qualidade das probas), acurácia (comunicação),
  **calibração** (reliability diagram — gerar e salvar a figura).
- **Baselines obrigatórios** (no mesmo split): taxa-base, sempre-mandante,
  maior-ranking. **Regra dura:** o modelo final **tem de** vencer a taxa-base em
  RPS — caso contrário não aprendeu nada e a entrega não está pronta.
- **Critério de escolha do modelo final:** menor **RPS de validação**; em empate
  técnico (diferença dentro do ruído entre folds), vence **calibração melhor** e,
  ainda empatado, o **modelo mais simples** (parcimônia — menos risco de overfit com
  poucos jogos por seleção).

### 7.F — Calibração de probabilidades

Aplicar **depois** de escolher o estimador, via `CalibratedClassifierCV` com
`cv` **temporal** (não embaralhar). Por que importa em 1X2: a saída alimenta
Brier/log-loss/RPS — probas descalibradas (super/subconfiança) perdem nessas
métricas mesmo com boa acurácia.

- **Isotônica** se houver dados suficientes (temos ~28k jogos ≥1993) — flexível,
  não-paramétrica.
- **Platt/sigmoide** se a fatia de calibração ficar pequena (menos propensa a
  overfit).
- A logística multinomial costuma já vir bem calibrada; o HistGB raramente — por
  isso **medir RPS antes/depois** e só manter a calibração se melhorar. A figura de
  calibração final vai para `docs/train/figuras/`.

### 7.G — Faltantes e desbalanceamento

- **Faltantes (41,5% sem ranking, pré-1993):** duas saídas legítimas, **decididas
  por modelo** — (a) **imputar** (mediana + indicador de ausência) para os
  lineares; (b) **HistGB aceita NaN nativo** (sem imputar). Recomendação: não
  descartar as linhas (perderia sinal de seleções com pouca história recente) — o
  `sample_weight` e a opção de reportar a fatia ≥1993 já endereçam a relevância. Se
  um candidato linear sofrer com a cauda antiga, **filtrar por `date>=1993` ou
  `sample_weight>ε` é decisão medida** (comparar com/sem), não default.
- **Desbalanceamento (empate 22,7%):** preferir `sample_weight` (já existe) +,
  para o linear, `class_weight="balanced"` **testado** (medir se ajuda o RPS — às
  vezes piora calibração). **Não** usar SMOTE/oversampling (distorce probabilidades
  e quebra a estrutura temporal). O empate é justamente o que o RPS cobra; a
  calibração na fatia de jogos equilibrados é o termômetro.

### 7.H — Aumento simétrico (espelhar mando/visitante)

Registrado pela `decisoes.md` como decisão do `02`. Como **63/72** jogos de 2026
são neutros, vale **testar** duplicar o `train` trocando home↔away (e invertendo o
alvo) para reduzir o viés de mando. **Recomendação:** tratar como experimento
medido, **não** default — a primeira defesa é o modelo aprender o peso de
`is_neutral` corretamente. Se adotado: espelhar **apenas o train**, **dentro do
fold de treino** (nunca no teste nem no predict), e cuidar para não duplicar peso.

## 8. Conjunto de preditores (exato)

Das 31 colunas do `features.parquet`, **entram como `X` (22):**

```
is_neutral, confed_home, confed_away,
elo_home, elo_away, elo_diff,
rank_home, rank_away, rank_diff, points_diff,
form_pts_home, form_pts_away, form_gf_home, form_gf_away, form_ga_home, form_ga_away,
rest_days_home, rest_days_away,
h2h_games, h2h_winrate_home, h2h_available, tournament_tier
```

*(22 colunas — congelar esta lista em `FEATURE_NAMES` e persistir em
`meta["feature_names"]`; as 9 restantes ficam de fora, abaixo.)*

**Ficam de fora de `X` (chaves, alvo, peso):**

| Coluna | Por que fora |
|---|---|
| `split` | controle de partição |
| `match_no` | chave do calendário (vai para a submissão no `03`, não é feature) |
| `date` | **eixo do split temporal** (E1), não preditor — usá-la como feature vaza tendência de calendário e quebra a generalização para 2026 |
| `home`, `away` | identidade do time; **não** one-hot (alta cardinalidade, overfit, e o sinal de força já está no Elo/ranking) |
| `target_outcome` | **alvo** |
| `home_score`, `away_score` | **alvo** (rota de gols); nunca preditores de 1X2 (vazariam o resultado) |
| `sample_weight` | peso de treino, **não** feature |

**Notas:** `elo_diff` é redundante com `elo_home−elo_away` (idem `rank_diff`); para
o **linear** os `*_diff` carregam o sinal mais limpo — manter ambos é inofensivo
(regularização cuida) e ajuda o HistGB a achar cortes. `points_home/away` **não**
existem no artefato (decisão do `01`: escala incomparável entre eras) — só
`points_diff`, correto. **Uso de `sample_weight`:** no `.fit` (§7.D), nunca como
coluna de `X`.

## 9. Validação do artefato e testes (E7)

**No `run_pipeline.py`** — adicionar `validate_model()` e registrá-lo em
`ARTIFACT_VALIDATORS["02_train.ipynb"]` (espelhando `validate_features`). Deve
checar, **sem** o gabarito:

1. `models/model.joblib` existe e desserializa (`joblib.load`).
2. Bundle tem `pipeline` e `meta` com `feature_names`, `classes`, `seed`.
3. `meta["classes"]` é exatamente `["home","draw","away"]` (ordem do RPS) **como
   conjunto e ordem**; e casa com `set(pipeline.classes_)`.
4. `meta["feature_names"]` ⊆ colunas de `features.parquet` (e é a lista esperada).
5. **Smoke de predição:** carregar as 72 linhas `predict`, `predict_proba`, e
   verificar shape `(72, 3)`, sem NaN, todas em `[0,1]` e **somando ~1** por
   linha. (Garante que o `03` vai funcionar.)

**Em `tests/` (novo `tests/test_train.py`, espelhando `test_features.py`):**

- **Contrato do bundle:** estrutura, chaves de `meta`, `seed==42`,
  `sklearn_version` presente.
- **Ordem das classes:** `meta["classes"] == ["home","draw","away"]`.
- **Reprodutibilidade:** re-treinar com `SEED` fixo dá `predict_proba` idêntico
  (ou treinar é caro → testar que duas chamadas de `predict_proba` no mesmo
  bundle são determinísticas).
- **Sanidade probabilística:** nas 72 linhas `predict`, probas somam ~1, sem NaN,
  shape `(72,3)`.
- **Bate a taxa-base:** num split temporal pequeno e rápido (fixture leve), o
  RPS do modelo ≤ RPS da taxa-base (o teste de fogo da skill, automatizado).
- **Não lê o gabarito:** grep em `02_train.ipynb` por `worldcup-2026-results`
  (igual ao `test_features.py`).
- **`run_pipeline` verde:** `validate_model()` passa.

> Os testes podem ser **caros** (treino). Espelhar a estratégia do
> `test_features.py`: fixture de sessão que gera o artefato se ausente, e marcar
> os testes pesados de forma que o CI rode `pytest` num orçamento aceitável (ex.:
> treinar uma vez por sessão; o teste de RPS-vs-baseline pode usar um subconjunto
> temporal pequeno).

## 10. Arquitetura de implementação (engenharia)

> Seção de **engenharia** (revisada pelo `python-ml-engineer`), não de ciência:
> concretiza *como* construir o que as §§6–9 decidiram, sem virar o código do
> notebook. Os trechos abaixo são **esqueletos verificados** na stack real (sklearn
> **1.7.1**, pandas **3.0.x**, numpy **2.3.x**, joblib **1.5.x**), não recortes para
> colar — preservam os guardrails das §3/§7.D. **Verificado** (com o
> `features.parquet` real): os 22 preditores existem com os dtypes esperados; os
> pipelines linear e HistGB abaixo dão `predict_proba` `(72, 3)` somando 1.

### 10.1 — Lista de features congelada e seleção de colunas

`FEATURE_NAMES` é a **fonte única** (vira `meta["feature_names"]`); derive os
subgrupos dela por **diferença de conjuntos**, nunca redigitando, para que nenhuma
coluna escape de um grupo silenciosamente:

```python
FEATURE_NAMES = [
    "is_neutral", "confed_home", "confed_away",
    "elo_home", "elo_away", "elo_diff",
    "rank_home", "rank_away", "rank_diff", "points_diff",
    "form_pts_home", "form_pts_away", "form_gf_home", "form_gf_away",
    "form_ga_home", "form_ga_away",
    "rest_days_home", "rest_days_away",
    "h2h_games", "h2h_winrate_home", "h2h_available", "tournament_tier",
]  # 22 — congelada; ordem == ordem em meta["feature_names"]

CAT_FEATURES  = ["confed_home", "confed_away"]          # string nullable
BOOL_FEATURES = ["is_neutral", "h2h_available"]         # boolean nullable
NUM_FEATURES  = [c for c in FEATURE_NAMES               # o resto é numérico
                 if c not in CAT_FEATURES + BOOL_FEATURES]
assert set(NUM_FEATURES + CAT_FEATURES + BOOL_FEATURES) == set(FEATURE_NAMES)
```

> **Guardrail de dtypes (verificado):** o parquet entrega `is_neutral`/
> `h2h_available` como **`boolean` nullable** e `confed_*` como **`string`
> nullable**. Testado de ponta a ponta: o `ColumnTransformer` linear e o HistGB
> consomem esses dtypes **direto do parquet** sem conversão manual — o
> `OneHotEncoder` e o `StandardScaler` aceitam o nullable, e o HistGB aceita
> `boolean` cru. **Não** faça `.astype(...)` defensivo fora do `Pipeline`: isso
> recria o risco de skew train↔serve (a mesma conversão teria de ser repetida no
> `03`). A única exceção é a rota "HistGB com categóricas nativas" (10.3), em que a
> conversão para `category` precisa estar **dentro** do `Pipeline` (num passo
> próprio), nunca aplicada solta ao DataFrame.

### 10.2 — Pipeline linear (LogisticRegression)

Pré-processamento e estimador no **mesmo** `Pipeline`, para o split ajustar tudo só
no treino do fold (§7.D). Imputação + indicador de ausência + scaler nas
numéricas; one-hot nas categóricas; booleanas convertidas para float por um passo
do próprio transformer:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression

num_lin = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),  # +sinal de ausência (pré-1993)
    ("scaler", StandardScaler()),
])
pre_lin = ColumnTransformer([
    ("num", num_lin, NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ("bool", "passthrough", BOOL_FEATURES),   # boolean nullable -> sklearn trata como 0/1
])
pipe_lin = Pipeline([
    ("pre", pre_lin),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])
```

Notas: (a) `add_indicator=True` é o que materializa o "sinal de época" dos 41,5% de
ranking ausente (§7.G) — sem ele, a mediana apaga a informação de que a linha é
pré-1993. (b) `random_state=SEED` na `LogisticRegression` só importa em solvers
estocásticos, mas custodiá-lo é barato e documenta a intenção. (c) o `passthrough`
de booleanas funciona porque o sklearn coage `boolean`→`{0,1}`; se algum dia surgir
NaN em `is_neutral`/`h2h_available` (hoje **0** no `train` e no `predict`), trocar
por um `SimpleImputer(strategy="most_frequent")` nesse grupo.

### 10.3 — Pipeline HistGB (cru, sem scaler/imputação)

O HistGB **aceita NaN nativamente** (resolve os 41,5% sem imputar) e dispensa
scaler. Duas variantes válidas — escolher por RPS no §6.5:

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Variante A — confed via one-hot (mesmo pré dos lineares, sem imputar/escalar):
pre_hgb = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
    remainder="passthrough",          # numéricas (com NaN) e booleanas passam cruas
)
pipe_hgb = Pipeline([
    ("pre", pre_hgb),
    ("clf", HistGradientBoostingClassifier(random_state=SEED)),
])

# Variante B (verificada, recomendada p/ o HistGB) — categóricas NATIVAS:
to_category = FunctionTransformer(
    lambda X: X.assign(**{c: X[c].astype("category") for c in CAT_FEATURES})
)
pipe_hgb_native = Pipeline([
    ("to_cat", to_category),          # dentro do Pipeline: sem skew
    ("clf", HistGradientBoostingClassifier(
        random_state=SEED, categorical_features="from_dtype")),
])
```

> **Determinismo (verificado):** dois `fit` do HistGB com `random_state=SEED` e o
> mesmo `sample_weight` produzem `predict_proba` **idêntico** (`np.allclose` True).
> Logo o teste de reprodutibilidade da §9 é factível para o HistGB. *Cautela de
> paralelismo:* se um dia adicionar `n_jobs`/threads, fixar a contagem (ou
> `OMP_NUM_THREADS`) para o determinismo se manter bit-a-bit; com o default atual
> não é problema.

### 10.4 — Harness walk-forward leak-free (o coração do §6.1)

Uma rotina única, usada por **todos** os candidatos **e** baselines (comparação
justa). Itera `TimeSeriesSplit` sobre o `train` **ordenado por data**, ajusta o
`Pipeline` só no treino do fold (o `ColumnTransformer` faz `fit` no treino e
`transform` no teste — o leak-free vem de o pré-processamento morar **dentro** do
`Pipeline`), passa `sample_weight` no `.fit` e **avalia sem peso** (imita os 72
jogos de produção, §7.D). RPS agregado como **média dos RPS por fold** (cada fold
pesa igual; é o número que casa com a leitura "acertaria a Copa X?").

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
# rps_mean / brier_multiclass / log_loss_mean / accuracy: replicados da skill,
# MESMA ordem OUTCOMES=("home","draw","away") (ver 10.6).

OUTCOMES = ("home", "draw", "away")

def _onehot(y):
    idx = np.array([OUTCOMES.index(o) for o in y])
    oh = np.zeros((len(y), 3)); oh[np.arange(len(y)), idx] = 1.0
    return oh

def _proba_in_outcome_order(fitted, X):
    """Reordena predict_proba de classes_ (alfabético) -> OUTCOMES (RPS)."""
    cols = [list(fitted.classes_).index(o) for o in OUTCOMES]
    return fitted.predict_proba(X)[:, cols]

def evaluate(estimator, X, y, w, dates, n_splits=5):
    order = np.argsort(dates.to_numpy(), kind="mergesort")   # ordena por tempo (estável)
    Xs, ys, ws = X.iloc[order], y.iloc[order], w.iloc[order]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for tr_idx, te_idx in tscv.split(Xs):
        model = clone(estimator)
        # sample_weight chega ao passo final do Pipeline via "clf__sample_weight"
        model.fit(Xs.iloc[tr_idx], ys.iloc[tr_idx],
                  clf__sample_weight=ws.iloc[tr_idx].to_numpy(dtype=float))
        proba = _proba_in_outcome_order(model, Xs.iloc[te_idx])
        oh = _onehot(ys.iloc[te_idx])
        rows.append({                              # avaliação SEM peso (§7.D)
            "rps": rps_mean(proba, oh),
            "brier": brier_multiclass(proba, oh),
            "log_loss": log_loss_mean(proba, oh),
            "acc": accuracy(proba, np.argmax(oh, axis=1)),
        })
    per_fold = pd.DataFrame(rows)
    return per_fold.mean().to_dict(), per_fold   # média entre folds + detalhe por fold
```

Pontos de engenharia que **evitam erro silencioso**:

- **Ordenar por data uma vez, fora do loop** (`mergesort` = estável: empates de data
  mantêm a ordem `(date, home)` do artefato — determinístico). O `TimeSeriesSplit`
  pressupõe ordem temporal; passar `X` na ordem do parquet **sem** reordenar é o
  modo mais fácil de vazar.
- **`clone(estimator)` por fold:** garante que nenhum estado fitado de um fold
  contamine o próximo (`fit` re-treina, mas `clone` é a rede de segurança explícita
  e barata).
- **`clf__sample_weight`:** o roteamento por nome de passo é o único jeito de o peso
  chegar ao estimador final do `Pipeline`. **Verificado** que funciona com `clf` como
  nome do passo. (Não usar a API de metadata routing/`set_fit_request` da 1.7 aqui:
  o prefixo `passo__sample_weight` no `.fit` é suficiente e mais legível.)
- **Reordenar `predict_proba` por nome** (`_proba_in_outcome_order`): `classes_` é
  alfabético `(away, draw, home)`; o RPS exige `(home, draw, away)`. Esta função é a
  **mesma** lógica que o `03` e a `validate_model` usam — centralizar evita três
  implementações divergentes do mapeamento (risco real, ver §11).
- **Fatia ≥1993 (§7.D):** reaplicar `evaluate` ao subconjunto `dates >= "1993-01-01"`
  e reportar à parte. É a fatia mais parecida com 2026; nenhuma mudança no harness,
  só no recorte de entrada.

**Baselines no mesmo split (§6.1):** computá-los **dentro do mesmo loop**, estimando
a taxa-base **só com `ys.iloc[tr_idx]`** (`value_counts(normalize=True)` reindexado
em `OUTCOMES`, broadcast para a matriz do teste). Estimá-la no histórico inteiro
vaza o futuro do fold. Sempre-mandante e maior-ranking comparam por acurácia (a skill
faz assim); para o RPS o baseline probabilístico é a taxa-base.

### 10.5 — Calibração temporal sem leak

`CalibratedClassifierCV` com `cv` **temporal** (`TimeSeriesSplit`), nunca `cv=int`
(que embaralha) nem `cv="prefit"` sobre um modelo já treinado em tudo (calibraria no
mesmo dado do fit → otimista). Verificado que `CalibratedClassifierCV(base, method=
..., cv=TimeSeriesSplit(k))` ajusta e expõe `classes_` corretamente, e que aceita
`sample_weight`:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    pipe_escolhido, method="isotonic", cv=TimeSeriesSplit(5)
)
# medir evaluate(calibrated, ...) vs. evaluate(pipe_escolhido, ...) e
# MANTER a calibração só se o RPS melhorar (§7.F).
```

> **Risco de leak na calibração (sinalizado — ver §11):** o ideal é que os folds da
> calibração interna **não** se sobreponham aos folds de validação externa do
> harness. Como ambos usam `TimeSeriesSplit` sobre o mesmo `train` ordenado, há
> sobreposição parcial. Mitigação pragmática: tratar o número do `evaluate` sobre o
> `CalibratedClassifierCV` como **estimativa levemente otimista** e confiar mais na
> comparação *relativa* (com vs. sem calibração, mesmo viés nos dois) do que no valor
> absoluto. O número honrado de qualidade continua sendo o do estimador não-calibrado
> no harness externo; a calibração é decidida por *delta* de RPS.

### 10.6 — Métricas: replicar a skill (não reimportar)

O `02` **não** deve importar de `worldcup-2026-results`-adjacentes, mas **pode**
importar as funções puras de métrica de
`.claude/skills/avaliar-previsoes/scripts/score_predictions.py` — elas não tocam o
gabarito (são só numpy sobre `probs`/`onehots`). O `test_features.py` já abre
precedente importando `DEFAULT_ALIASES` de lá via `sys.path`. **Recomendação:**
replicar `rps_mean`, `brier_multiclass`, `log_loss_mean`, `accuracy` no notebook
(curtas, ~4 linhas cada) **copiando a matemática exata** e mantendo `OUTCOMES=
("home","draw","away")`, e cobrir com um teste de paridade (10.8) que compara contra
a versão da skill num vetor fixo. Assim o número do `02` casa bit-a-bit com a
avaliação final, sem acoplar o pipeline à skill.

### 10.7 — Serialização do bundle (contrato exato)

Um único `joblib.dump` de um dict `{"pipeline", "meta"}` (§2.1). Esqueleto do
`meta` e da escrita, já com os campos que a `validate_model` vai cobrar:

```python
import joblib, sklearn
from datetime import datetime, timezone

meta = {
    "feature_names": list(FEATURE_NAMES),        # ordem exata de entrada do 03
    "classes": list(OUTCOMES),                   # ["home","draw","away"] (ordem do RPS)
    "model_name": "logreg_isotonic",             # rótulo do escolhido
    "seed": SEED,                                # 42
    "sklearn_version": sklearn.__version__,      # guardrail de compatibilidade
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "cv_rps": float(cv_rps_do_escolhido),        # RPS médio walk-forward (sanity)
}
bundle = {"pipeline": pipe_final_fitted, "meta": meta}
joblib.dump(bundle, MODELS / "model.joblib")
```

Contrato (o que o `03`/`validate_model`/testes assumem):

- `bundle["pipeline"]` está **fitted no `train` inteiro** (com `sample_weight`) e
  expõe `predict_proba`. Pode ser o `Pipeline` cru **ou** o `CalibratedClassifierCV`
  que o envolve — em ambos `predict_proba` e `classes_` existem.
- `meta["classes"] == ["home","draw","away"]` e **`set(meta["classes"]) ==
  set(pipeline.classes_)`** (a ordem do `meta` é a do RPS; a do `classes_` é
  alfabética — por isso o `03` mapeia **por nome**, §2.1).
- `meta["feature_names"]` é a lista de 22 colunas e é `⊆` das colunas do
  `features.parquet`.
- `meta["created_at"]` é timezone-aware (UTC) — `datetime.now(timezone.utc)`,
  **não** o `utcnow()` (deprecado na 3.12+). É metadado, não entra no fuso GMT-3 do
  calendário.

### 10.8 — `validate_model()` no `run_pipeline.py` (mesmo estilo de `validate_features`)

Espelhar `validate_features`: função pura, levanta `ValidationError`, devolve um
`int` (ex.: nº de linhas previstas) e é registrada em `ARTIFACT_VALIDATORS`. Para
reusar o smoke de predição **sem** reabrir o parquet à mão, ela carrega as 72 linhas
`predict` do `FEATURES_PATH` já conhecido pelo módulo. Esqueleto:

```python
MODEL_PATH = ROOT / "models" / "model.joblib"
EXPECTED_CLASSES = ["home", "draw", "away"]

def validate_model(model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH) -> int:
    if not model_path.is_file():
        raise ValidationError(f"Artefato ausente: {model_path}")
    bundle = joblib.load(model_path)                       # desserializa
    if not isinstance(bundle, dict) or {"pipeline", "meta"} - bundle.keys():
        raise ValidationError("model.joblib: esperado dict com 'pipeline' e 'meta'")
    pipe, meta = bundle["pipeline"], bundle["meta"]
    for key in ("feature_names", "classes", "seed"):
        if key not in meta:
            raise ValidationError(f"model.joblib: meta sem '{key}'")
    if list(meta["classes"]) != EXPECTED_CLASSES:
        raise ValidationError(f"model.joblib: classes != {EXPECTED_CLASSES}")
    if set(meta["classes"]) != set(pipe.classes_):
        raise ValidationError("model.joblib: meta['classes'] != pipeline.classes_")
    feats = list(meta["feature_names"])
    features = pd.read_parquet(features_path)
    missing = set(feats) - set(features.columns)
    if missing:
        raise ValidationError(f"model.joblib: feature_names fora do parquet: {sorted(missing)}")
    predict = features[features["split"] == "predict"]
    proba = pipe.predict_proba(predict[feats])             # smoke nas 72 linhas
    if proba.shape != (len(predict), 3):
        raise ValidationError(f"model.joblib: predict_proba shape {proba.shape}")
    if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
        raise ValidationError("model.joblib: probabilidades inválidas (NaN/fora de [0,1])")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValidationError("model.joblib: probabilidades não somam 1")
    return len(predict)

ARTIFACT_VALIDATORS = {
    "01_features.ipynb": validate_features,
    "02_train.ipynb": validate_model,     # <- acrescentar esta linha
}
```

> **Importes no topo do `run_pipeline.py`:** `validate_model` adiciona `import joblib`
> e `import numpy as np` (hoje o módulo só importa `pandas`/`nbformat`/`nbclient`).
> `joblib` é dependência transitiva do sklearn, mas **declará-lo em `requirements.txt`**
> é o correto (não depender de transitividade). Confirmar que `joblib` está no
> `requirements.txt`; se não, é um item de tarefa (ver `tarefas.md`).

### 10.9 — `tests/test_train.py` (mesmo padrão de `test_features.py`)

Espelhar a estrutura: fixture de sessão que **gera o artefato se ausente** (executa
o `02` via `nbclient`, igual ao `features_df`), e testes pequenos sobre o bundle. O
treino do `02` pode ser pesado; mitigar como na §9:

- A fixture `model_bundle` (escopo de sessão) faz `joblib.load`; se o arquivo não
  existir, executa o notebook uma vez. Reúso máximo entre testes.
- **Contrato/ordem/sanidade/grep** são baratos (só leem o bundle e o parquet) — rodam
  sempre.
- **RPS ≤ taxa-base** e **determinismo** são os custosos. Para o RPS ≤ taxa-base,
  treinar um pipeline **leve** (ex.: `LogisticRegression`) numa **fatia temporal
  pequena** dentro do próprio teste, com um único corte treino/holdout — não reusar o
  walk-forward de 5 folds do notebook. Determinismo: comparar **duas chamadas de
  `predict_proba` no mesmo bundle** (sempre idênticas) é o teste barato e suficiente;
  o re-treino bit-a-bit fica como teste opcional marcado.
- **Grep do gabarito:** idêntico ao `test_features.py`, mas sobre `02_train.ipynb` —
  asserta que `"worldcup-2026-results"` não aparece em nenhuma célula de código.
- **Paridade de métricas (10.6):** comparar as funções replicadas no notebook contra
  `score_predictions.rps_mean`/etc. num vetor fixo (`approx`), garantindo que o `02`
  mede como a skill.

## 11. Riscos de engenharia e vazamento (a vigiar)

Itens que o plano de ciência não cobre explicitamente e que **quebram em silêncio**
se ignorados. Cada um tem mitigação acionável (refletida em `tarefas.md`).

| # | Risco | Por que morde | Mitigação |
|---|---|---|---|
| R1 | **Ordem das classes** (`classes_` alfabético `away,draw,home` ≠ RPS `home,draw,away`) | mapear por **posição** inverte home↔away e o RPS fica péssimo sem erro visível | função única `_proba_in_outcome_order` reusada em harness/03/validate; `meta["classes"]`; teste de ordem (§9) |
| R2 | **`multi_class` removido na 1.7** | `requirements.txt` fixa `scikit-learn>=1.4,<2` — esse intervalo **cruza a fronteira da remoção**: em 1.4 `multi_class="multinomial"` ainda funciona (silencioso), em 1.7 some. Um notebook que dependa dele passa na máquina do autor e **quebra no CI** | assinatura **sem** `multi_class` (já corrigido §6.4/§7.B); multinomial é o default. Considerar apertar o piso do pin p/ ≥1.6 num passo futuro |
| R3 | **Leak da calibração** (folds internos do `CalibratedClassifierCV` sobrepõem os do harness) | RPS pós-calibração levemente otimista; decisão "calibrar?" pode sair errada | decidir por **delta** de RPS (mesmo viés nos dois); valor absoluto vem do estimador não-calibrado (§10.5) |
| R4 | **`sample_weight` não chega ao estimador** | sem o prefixo `clf__sample_weight`, o peso é ignorado **sem erro** e o modelo treina como se tudo pesasse 1 | roteamento `passo__sample_weight` verificado; um teste pode checar que pesos extremos mudam a saída |
| R5 | **Avaliar COM peso por engano** | RPS ponderado não é o da skill (que pontua os 72 sem peso) → modelo escolhido para a métrica errada | `evaluate` avalia **sem** peso (§7.D/§10.4); peso só no `.fit` |
| R6 | **`fit` no dataset inteiro antes de dividir** (scaler/imputer/one-hot global) | vaza estatística do futuro para o passado; "acurácia suspeita" | pré-processamento **dentro** do `Pipeline`; nunca `fit_transform` global; `clone` por fold |
| R7 | **Consistência `feature_names` 01↔02↔03** | se o `01` mudar nome/ordem de coluna, o `03` seleciona `predict[feats]` e **falha** (ou pior, silencia com `handle_unknown`) | `validate_model` checa `feature_names ⊆ parquet`; `FEATURE_NAMES` derivado da §8; o `03` seleciona por `meta["feature_names"]` |
| R8 | **Coluna faltando em produção** | `KeyError` no `03` ou zeros silenciosos do `OneHotEncoder(handle_unknown="ignore")` | smoke de `predict_proba` nas 72 linhas dentro de `validate_model` (falha cedo, no pipeline) |
| R9 | **Skew de dtype nullable** | se alguém `.astype()` fora do `Pipeline` no `02`, o `03` precisa repetir e diverge | **proibir** conversão fora do `Pipeline` (§10.1); verificado que sklearn consome o nullable cru |
| R10 | **`sklearn_version` divergente ao desserializar** | bundle salvo em 1.7.1 e carregado noutra minor pode emitir `InconsistentVersionWarning` ou quebrar | persistir `meta["sklearn_version"]`; `validate_model` pode **avisar** (não falhar) se diferir do runtime |
| R11 | **Performance/custo de CI** | walk-forward × candidatos × calibração sobre 49k linhas pode estourar o timeout do `pytest`/`run_pipeline` | LogReg e HistGB são rápidos nessa escala; testes pesados em fatia pequena (§10.9); `run_pipeline --timeout` já existe; medir antes de otimizar |
| R12 | **Determinismo do HistGB sob threads** | paralelismo pode quebrar reprodutibilidade bit-a-bit | default é determinístico (verificado); se ligar `n_jobs`, fixar threads/seed |

> **Nota ao cientista (não reverte decisão, registra divergência leve):** sobre
> **avaliar sem peso** (§7.D) — concordo que imita produção e é o default certo. Um
> contraponto de engenharia a registrar: como o `train` mistura eras com
> `sample_weight` de 0 a 2,42, o RPS *não-ponderado* dos folds antigos pode dominar a
> média e penalizar um modelo que é ótimo na era recente. A **fatia ≥1993** já
> mitiga isso ao dar a leitura "parecida com 2026"; sugiro que a **seleção final
> priorize o RPS da fatia ≥1993** (não o RPS global), mantendo o global como
> contexto. É refinamento de leitura, não mudança de método.

## 12. Mapeamento para o notebook

Hoje o `02` tem: setup (`find_root`, caminhos, `MODELS.mkdir`), carga de
`features.parquet`, um markdown "Escolha e treine o modelo" e um `# TODO` de
`joblib.dump`. O plano:

- **Manter** o setup e a carga.
- **Inserir** as seções §6.0–§6.7 (cada uma com cabeçalho markdown + células),
  **na ordem**: harness/baselines (§6.1) **antes** dos modelos (a régua primeiro),
  depois baseline forte, alternativas, calibração+seleção, persistência.
- **Substituir** o `# TODO` final pela escrita do bundle
  `{"pipeline","meta"}` em `models/model.joblib`.
- Salvar a tabela de resultados (modelo × métricas vs. baselines) e a figura de
  calibração em `docs/train/figuras/` (registro versionado, como a EDA).

## 13. Critérios de conclusão (Definition of Done)

- [ ] `models/model.joblib` é um bundle `{"pipeline","meta"}`; `pipeline` está
      *fitted* e dá `predict_proba` em `(72,3)`.
- [ ] `meta` tem `feature_names`, `classes=["home","draw","away"]`, `seed=42`,
      `model_name`, `sklearn_version`.
- [ ] Validação **walk-forward** (sem K-fold aleatório); pré-processamento só no
      treino do fold; `sample_weight` no fit.
- [ ] Tabela **modelo × {RPS, Brier, log-loss, acc}** vs. os **3 baselines**; o
      modelo final **vence a taxa-base em RPS**.
- [ ] Calibração medida (antes/depois) e figura salva em `docs/train/figuras/`.
- [ ] Decisões registradas: alvo (1X2), tratamento de NaN, peso na
      avaliação, (se usado) aumento simétrico.
- [ ] `validate_model()` no `run_pipeline.py` + testes em `tests/test_train.py`;
      `ruff` + `pytest` + `run_pipeline` verdes.
- [ ] Nenhuma célula lê `worldcup-2026-results.csv`.
- [ ] `03` ajustado para carregar o bundle e escrever a submissão no contrato da
      skill (`match, home, away, p_home, p_draw, p_away`).

> Tarefas acionáveis: [`tarefas.md`](tarefas.md).
