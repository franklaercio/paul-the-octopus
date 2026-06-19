# Plano de Execução — Features (`notebooks/01_features.ipynb`)

Plano da etapa de engenharia de features do Paul the Octopus. O `01` é a **única
etapa onde a ciência das features vive**; `02` (treino) e `03` (inferência) só
consomem o que ele grava. Produto: `data/processed/features.parquet`.

Pré-requisito de leitura: a EDA ([`docs/eda/README.md`](../eda/README.md)) já
mapeou qualidade, chaves e os sinais mais fortes. Este plano transforma aqueles
achados em features, sem refazer a exploração.

> **Decisão central deste plano:** o `features.parquet` contém **as duas
> populações** — linhas de **treino** (cada partida histórica, com alvo) e linhas
> de **previsão** (os 72 jogos de 2026, sem alvo) — distinguidas por uma coluna
> `split`. Assim a *mesma* lógica de feature serve treino e inferência (sem
> *train/serve skew*), e o `03` lê `features.parquet` em vez de recomputar tudo.

---

## 1. Objetivos

1. Produzir um `features.parquet` **sem vazamento temporal**: cada feature usa só
   informação disponível **antes** da partida.
2. Entregar as features de **maior sinal** primeiro (Elo e ranking, conforme a
   EDA), de forma vetorizada e reprodutível.
3. Garantir **consistência treino↔inferência**: uma só rotina computa as features
   para o histórico e para os 72 jogos de 2026.
4. Manter-se **agnóstico ao modelo**: gravar tanto o alvo 1X2 quanto os placares,
   para não fechar a porta de um modelo de gols (Poisson/Dixon-Coles).
5. Acrescentar **validação do artefato** ao `run_pipeline.py` e **testes** ao
   `tests/` (o CI roda `ruff` + `pytest`), como manda a arquitetura.

## 2. Escopo, fontes e contrato de saída

**Fontes (entrada):** `historical-results.csv`, `ranking.csv`,
`matches-schedule.csv` e, opcionalmente, `shootouts.csv` /
`historical_win-loose-draw_ratios.csv`. **Nunca** `worldcup-2026-results.csv`
(gabarito; as previsões dos 72 jogos são feitas a partir do estado **pré-torneio**,
sem realimentar resultados do Mundial).

**Contrato de `features.parquet`** (uma linha por partida):

| Coluna | Tipo | Aplica a | Descrição |
|---|---|---|---|
| `split` | str | ambos | `train` (histórico) ou `predict` (72 jogos 2026) |
| `match_no` | int? | predict | número do jogo no calendário (chave p/ o `03`) |
| `date` | date | ambos | data da partida |
| `home`, `away` | str | ambos | seleções já **canônicas** (mapa da skill) |
| `is_neutral` | bool | ambos | campo neutro? (define se vale o mando) |
| `confed_home`, `confed_away` | str | ambos | confederação |
| `elo_home`, `elo_away`, `elo_diff` | float | ambos | Elo **pré-jogo** |
| `rank_home`, `rank_away`, `rank_diff`, `points_diff` | float | ambos | ranking FIFA *as-of* (≥1993; NaN antes) |
| `form_pts_*`, `form_gf_*`, `form_ga_*` | float | ambos | forma recente (janela móvel, deslocada) |
| `rest_days_home`, `rest_days_away` | float | ambos | dias desde o último jogo |
| `h2h_winrate_home`, `h2h_games`, `h2h_available` | float/bool | ambos | confronto direto (esparso) |
| `target_outcome` | str | train | `home`/`draw`/`away` (alvo 1X2) |
| `home_score`, `away_score` | int | train | placar (rota de modelo de gols) |
| `sample_weight` | float | train | peso (importância do torneio × recência) |

O alvo final do projeto (definido pela skill `avaliar-previsoes`) é a probabilidade
1X2 dos 72 jogos; este contrato sustenta tanto um classificador 1X2 direto quanto
um modelo de gols.

> **Ajuste necessário no `03`:** acrescentar `features.parquet` às suas entradas
> (filtrando `split == "predict"`) e parar de depender só do calendário cru.

## 3. Princípios e guardrails (anti-vazamento)

O maior risco da etapa. Regras por feature:

- **Elo:** registrar o valor **antes** do jogo e atualizar **depois**, percorrendo
  o histórico em ordem cronológica. Mando (HFA) só quando `is_neutral == False`.
- **Janelas móveis (forma, força):** `shift(1)` dentro da série de cada seleção —
  a linha nunca enxerga o próprio jogo.
- **Ranking *as-of*:** `merge_asof` pegando o último ranking **estritamente antes**
  da data da partida.
- **Reconciliação de nomes:** reutilizar o `DEFAULT_ALIASES` de
  `.claude/skills/avaliar-previsoes/scripts/score_predictions.py` (fonte única).
- **Limpeza herdada da EDA:** deduplicar o 1 confronto duplicado
  (`date+home+away`); tratar os 21 `rank` nulos do ranking.
- **Pré-requisito de ambiente:** garantir `pyarrow` instalado (o `features.parquet`
  depende dele) — `pip install -r requirements-dev.txt`.
- **Reprodutibilidade:** semente fixa, operações determinísticas, vetorizadas.

### Codificação e escalonamento — onde fica

Para evitar vazamento e manter consistência treino↔inferência:

- **Escalonamento de features numéricas** (z-score/MinMax): **não** no `01`. O
  `features.parquet` guarda os valores crus; o *fit* do scaler vai no `02`, dentro
  de um `Pipeline` do scikit-learn (ajustado só no treino de cada *fold*). Modelos
  de árvore/boosting e Elo/Poisson nem exigem escalonamento — por isso é decisão do
  `02`/modelo, não do `01`.
- **Categóricas nominais** (ex.: `tournament`, ~200 valores): **não** rotular como
  `0,1,2…` cru — isso inventa uma ordem falsa que prejudica modelos lineares/SVM/NN.
  Em vez disso: (a) **reduzir a cardinalidade** para poucos baldes de domínio
  (amistoso, eliminatória continental, torneio continental, eliminatória de Copa,
  Copa do Mundo, outros) e fazer **one-hot** no `Pipeline` do `02`; e/ou (b) mapear
  para um **ordinal de importância** — aí a ordem é real (amistoso < eliminatória <
  torneio principal) — útil como feature e como base do `sample_weight`. O
  *bucketing*/ordinal é um mapa fixo de domínio e pode ficar no `01`; o one-hot via
  encoder fica no `02`. (Em 2026 todos os 72 jogos caem no balde "Copa do Mundo".)
- **Nomes de seleção:** validados — só Curaçao, Irã e Coreia do Sul tinham grafias
  múltiplas (todas unificadas pelo `canon`); França é só "France". Manter o mapa
  único da skill como fonte de verdade.

## 4. Plano geral (etapas)

| Etapa | Seção | Entrega |
|---|---|---|
| **E0** Prep, ambiente e reconciliação | §6.0 | tabela longa limpa + nomes canônicos |
| **E1** Núcleo de força: Elo + ranking *as-of* | §6.1 | features Tier 1 |
| **E2** Mando/neutro e contexto 2026 | §6.2 | `is_neutral` + mando dos anfitriões |
| **E3** Forma e força ofensiva/defensiva | §6.3 | features Tier 2 |
| **E4** Descanso/congestionamento | §6.4 | `rest_days_*` |
| **E5** Confronto direto e confederação | §6.5 | features Tier 3 |
| **E6** Montagem do `features.parquet` | §6.6 | artefato (train+predict) + contrato |
| **E7** Validação e testes | §6.7 | checagem no `run_pipeline` + `pytest` |

## 5. Tópicos

- **Consistência treino↔inferência** (a mesma rotina para histórico e 2026).
- **Anti-vazamento** (Elo/forma/ranking calculados só com o passado).
- **Priorização por sinal** (Elo e ranking primeiro; resto incremental).
- **Agnosticismo ao modelo** (alvo 1X2 + placares).
- **Tratamento de campo neutro** (decisivo em 2026).
- **Qualidade e cobertura** (dedupe, nulos, ranking só ≥1993, H2H esparso).

## 6. Seções e transformações (detalhado)

### §6.0 — Prep, ambiente e reconciliação (E0)
Carregar fontes; aplicar `canon()` (mapa da skill) a todas as seleções; deduplicar;
montar uma **tabela longa por seleção-jogo** (uma linha por time por partida, em
ordem cronológica) — a base para Elo, forma e descanso. **Saída:** base limpa.

### §6.1 — Núcleo de força: Elo + ranking *as-of* (E1)
São o sinal mais forte da EDA (P(vitória mandante) variou 0,07→0,80 por Elo e
0,15→0,80 por ranking). Calcular Elo pré-jogo (com HFA condicionado a `is_neutral`)
e juntar o ranking *as-of*; derivar `*_diff`. **Saída:** Tier 1.

### §6.2 — Mando/neutro e contexto 2026 (E2)
A EDA mostrou que campo neutro derruba a vitória do mandante (0,507→0,442). Definir
`is_neutral` no histórico e, para 2026, dar mando só aos anfitriões (EUA/Canadá/
México) e neutro aos demais. **Saída:** mando coerente entre treino e previsão.

### §6.3 — Forma e força ofensiva/defensiva (E3)
Janelas móveis (deslocadas) de pontos e gols pró/contra por seleção; opção de
ponderar por recência. Base também para um modelo de gols. **Saída:** Tier 2.

### §6.4 — Descanso/congestionamento (E4)
Dias desde o último jogo de cada seleção (da tabela longa). Para 2026, calcular a
partir do calendário. **Saída:** `rest_days_*`.

### §6.5 — Confronto direto e confederação (E5)
H2H a partir de `historical_win-loose-draw_ratios.csv` (ou computado), com flag de
disponibilidade — **esparso (14/72)**, usar com cautela. Confederação como
categórica. **Saída:** Tier 3.

### §6.6 — Montagem do `features.parquet` (E6)
Unir as features das duas populações (`train` e `predict`), anexar alvo e placares
(treino), `sample_weight` e `split`; gravar conforme o contrato (§2). **Saída:**
`features.parquet`.

### §6.7 — Validação e testes (E7)
Acrescentar ao `run_pipeline.py` a validação do artefato (schema/dtypes; sem NaN nas
chaves; **72** linhas `predict` cobrindo todo o calendário; alvo presente sse e só
se `split=="train"`). Adicionar testes em `tests/` (inclusive um *smoke test* de
vazamento). **Saída:** pipeline e CI verdes.

## 7. Features priorizadas (da EDA)

| Tier | Feature | Por quê |
|---|---|---|
| **1** | `elo_diff`, `rank_diff`/`points_diff` | sinal mais forte e barato |
| **1** | `is_neutral` (+ mando 2026) | muda o significado do mando |
| **2** | forma recente (pts/gols, deslocada) | capta momento |
| **2** | força ofensiva/defensiva | base p/ modelo de gols |
| **2** | `rest_days_*` | fadiga/calendário |
| **3** | H2H (com flag) | esparso — cautela |
| **3** | confederação | contexto/estilo |

## 8. Decisões em aberto (com recomendação)

- **Alvo:** 1X2 *e* placares no artefato → **manter ambos** (agnóstico).
- **Orientação das linhas:** *home-oriented* simples vs. **aumento simétrico**
  (espelhar mando/visitante) — recomendado avaliar o simétrico por causa dos jogos
  neutros; deixar como decisão do `02`.
- **Janela de treino:** todo o histórico para o Elo, mas **ponderar/filtrar
  amistosos** (dominam o histórico) e priorizar era recente via `sample_weight`.
- **Ranking pré-1993:** ausente → deixar NaN e tratar no modelo, ou restringir as
  features de ranking ao período ≥1993.

## 9. Mapeamento para o notebook

Hoje o `01` tem: setup, carga só do histórico e um *pass-through*
(`features = historico.copy()`). O plano: estender a carga (ranking, calendário,
auxiliares), inserir as seções §6.0–§6.7 e **substituir o pass-through** pela
escrita do `features.parquet` no novo contrato.

## 10. Critérios de conclusão (Definition of Done)

- [ ] `features.parquet` com `split` `train`+`predict`; **72** linhas `predict`
      cobrindo todo o calendário; alvo só em `train`.
- [ ] Tier 1 (Elo + ranking + neutro) implementado e sem vazamento.
- [ ] Nenhuma feature lê o gabarito; checagens de vazamento passam.
- [ ] Validação do artefato no `run_pipeline.py` + testes em `tests/` verdes.
- [ ] Contrato (§2) documentado e `03` ajustado para ler `features.parquet`.

> Tarefas acionáveis: [`tarefas.md`](tarefas.md).
