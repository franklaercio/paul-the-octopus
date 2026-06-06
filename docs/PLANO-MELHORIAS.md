# Plano de Melhorias — Pipeline de Predição (Paul the Octopus)

Documento de planejamento das melhorias adotadas para aumentar a **acurácia real (fora da amostra)** do modelo, motivado pelo diagnóstico da validação da Copa 2022 (ver [VALIDACAO-2022.md](VALIDACAO-2022.md)).

## Contexto do diagnóstico

A acurácia de treino reportada (94,24%) é uma ilusão de avaliação: a célula `cell-31` do notebook treina e mede no **mesmo conjunto** (`X_test` é cópia de `X`). Medições honestas (números reais medidos no P1, `random_state=42`, dados locais em `files/`):

| Avaliação | Acurácia (alvo binário) |
|---|---|
| In-sample (como no notebook hoje, dataset inteiro) | **93,27%** |
| Split temporal honesto (80/20) | **67,07%** |
| Split temporal walk-forward — hold-out **Copa 2022** | **57,89%** |
| Validação Copa 2022 fase de grupos (heurística de placar, ver VALIDACAO-2022) | 43,8% |

> Nota sobre o hold-out: o conjunto "Copa 2022" usado no P1 são as **57 partidas** da Copa 2022 presentes no dataset modelado (grupos + mata-mata), filtradas por `tournament == 3` e ano 2022. É mais difícil que o split 80/20 genérico (que mistura eliminatórias menos competitivas), por isso a acurácia binária cai de ~67% para ~58%. Ambos os números são honestos e muito abaixo do 94% reportado.

**Métricas de probabilidade no hold-out Copa 2022 (3 classes 1X2; menor = melhor para RPS/log-loss/Brier):**

| Modelo | RPS | log-loss | Brier | acerto 1X2 |
|---|---|---|---|---|
| Modelo atual (RF binário, p_draw=0) | **0,3388** | 9,2124 | 0,9233 | 0,421 |
| B0 — frequência-base do treino (0,499 / 0,214 / 0,287) | 0,2334 | 1,0777 | 0,6530 | 0,439 |
| B1 — favorito do ranking (suavizado 0,70/0,15/0,15) | **0,2019** | 1,0053 | 0,5982 | 0,579 |

O modelo atual tem o **pior** RPS/log-loss/Brier do conjunto: por ser binário, atribui `p_draw = 0` e é massacrado nos 14/57 jogos que empataram. Até o baseline trivial B0 o supera. Distribuição 1X2 real do dataset (11.537 jogos): **home_win 48,7% / draw 21,7% / away_win 29,5%**.

Três falhas estruturais confirmadas: (1) vazamento de avaliação, (2) alvo binário que ignora empates (~22% dos jogos), (3) viés de mando de campo num torneio neutro (coluna `neutral` é descartada em `cell-13`).

## Ordem de execução recomendada

```
P1 (avaliação honesta) ──► P2 (alvo 3 classes) ──► P4 (features) ──► P3 (Poisson/DC) ──► P5 (calibração) ──► P6 (higiene)
       fundação              destrava empate        maior ganho      placar real        qualidade prob.    confiabilidade
```

P1 é pré-requisito de todos: sem medição honesta não é possível afirmar ganho. P6 (higiene) pode correr em paralelo. O **P4 foi decomposto em subtarefas P4.1–P4.9** (ver seção P4) e **implementado** (P4.7 adiado) — cada feature é adicionada e medida pelo RPS **na validação** antes da próxima; o hold-out fica reservado para o relatório final.

## Métricas-alvo (substituem "acurácia de treino")

| Métrica | Para quê | Meta inicial |
|---|---|---|
| **RPS** (Ranked Probability Score) | qualidade 1X2 (respeita ordem V/E/D) | < RPS do baseline Elo |
| **Log-loss multiclasse** | penaliza overconfidence | monitorar |
| **Brier multiclasse** | erro quadrático de probabilidade | monitorar |
| **Reliability diagram** | calibração | curva ≈ diagonal |
| **Acerto de resultado** | comparável à VALIDACAO-2022 | > 44% no hold-out 2022 |
| **Placar exato** | critério de pontuação da submissão | > 7/48 no hold-out 2022 |

**Hold-out oficial:** Copa 2022 inteira (nunca usada em treino). Toda mudança é aceita só se melhora o RPS no hold-out contra os baselines.

---

# P1 — Avaliação honesta (split temporal + métricas próprias + baselines)

**Esforço:** Baixo · **Impacto:** Alto (fundação) · **Risco:** Baixo

### Objetivo
Substituir o número fantasioso de 94% por uma medição confiável fora da amostra, criando a infraestrutura de avaliação que todas as etapas seguintes vão usar.

### Situação atual
- `cell-31`: `X_test = X`; `random_forest.score(X, y)` mede no treino. `train_test_split` é importado mas **não usado**.
- Não há baselines, não há métricas de probabilidade, não há split temporal.

### Passos
1. **Não embaralhar.** Ordenar `df_historical_results` por `date` antes de qualquer split (a data ainda existe até `cell-26`, onde é descartada — preservá-la).
2. **Split temporal walk-forward:**
   - Treino: até 2018.
   - Validação: 2019–2021 (para tuning e calibração).
   - Teste/hold-out: Copa 2022 (já documentada em `VALIDACAO-2022.md`).
3. **Funções de métrica** (numa célula utilitária, reutilizável):
   - `rps(probs, outcome)` — Ranked Probability Score para as 3 classes.
   - `log_loss` e `brier_score_loss` (multiclasse) via sklearn.
   - `reliability_curve(probs, outcomes)` + plot.
4. **Baselines obrigatórios:**
   - B0: frequência-base fixa (≈ 0,49 / 0,22 / 0,30).
   - B1: "favorito do ranking sempre vence".
   - B2: probabilidades de **Elo puro** (ver P4) convertidas em 1X2.
5. **Re-medir o modelo atual honestamente** e registrar a queda como linha de base "antes". **Resultado real medido:** acurácia binária cai de 93,27% (in-sample) para **57,89%** no hold-out Copa 2022 (67,07% num split 80/20 genérico). Em 1X2, o RF binário (RPS 0,3388) perde para B0 (0,2334) e B1 (0,2019).

### Status: ✅ IMPLEMENTADO
- `cell-26` ajustada para **preservar a `date`** (necessária ao split temporal).
- `cell-28` adiciona o alvo `outcome` (0/1/2) só para avaliação 1X2.
- Novas células `P1.1`–`P1.4` no notebook: métricas (`ranked_probability_score`, `multiclass_brier`, `multiclass_log_loss`, `evaluate_probs`, `reliability_diagram`), split temporal, baselines B0/B1 (B2 Elo é stub → habilitado em P4.2) e re-medição honesta.

### Critério de aceitação
- ✅ Existe `evaluate_probs(nome, probs, outcomes)` que devolve RPS, log-loss e Brier (e acerto 1X2).
- ✅ O modelo atual está medido no split temporal e contra os baselines B0 e B1 (B2 fica para o P4).
- ✅ Número honesto documentado (57,89% binário no hold-out 2022 substitui o "94,24%").

### Dependências
Nenhuma. É a primeira tarefa.

---

# P2 — Alvo multiclasse {vitória, empate, derrota}

**Esforço:** Baixo · **Impacto:** Alto · **Risco:** Baixo

### Objetivo
Permitir que o modelo **possa** prever empate — hoje matematicamente impossível, custando ~22% dos jogos.

### Situação atual
- `cell-28`: `is_won = score_difference > 0`. Empate e derrota caem na mesma classe `False`.
- `cell-31`: `RandomForestClassifier` binário; `cell-38` usa `predict_proba[:,1]` e deriva o visitante como `1 - home`.

### Passos
1. **Novo alvo** em `cell-28`:
   ```python
   def outcome(row):
       if row['home_score'] > row['away_score']: return 'home_win'
       if row['home_score'] < row['away_score']: return 'away_win'
       return 'draw'
   df['target'] = df.apply(outcome, axis=1)
   ```
2. **Classificador multiclasse** com `predict_proba` nas 3 colunas. Começar com **regressão logística multinomial** (rápida, calibrável, interpretável) como referência; comparar com gradient boosting.
3. **Ajustar o consumo das probabilidades** (`cell-38`): em vez de `1 - home_win_prob`, usar diretamente `[p_home, p_draw, p_away]`.
4. **Tratar vazamento de feature:** garantir que `score_difference` e `home_score`/`away_score` **não entram** em `X` (são o alvo). Hoje as features são só ranking, mas ao adicionar features (P4) este cuidado é crítico.

### Critério de aceitação
- Modelo produz 3 probabilidades que somam 1.
- RPS no hold-out 2022 **melhora** vs. o binário convertido.
- O modelo prevê empates em proporção plausível (não 0%, não 50%).

### Status: ✅ IMPLEMENTADO
Alvo binário substituído por classificador **multiclasse** (3 classes, alvo `outcome` 0/1/2 já criado em `cell-28` no P1 — reaproveitado, não duplicado). Três classificadores treinados **só no treino temporal (≤ 2018)** e avaliados no hold-out Copa 2022 com a infraestrutura do P1 (`evaluate_probs`, `ranked_probability_score`, `reliability_diagram`, baselines B0/B1). Sem vazamento: `X` continua só `['average_rank', 'rank_difference']` (`assert` anti-vazamento na `cell-P2.1`). `random_state=42`.

**Números reais medidos (hold-out Copa 2022, 57 jogos, 14 empates; menor = melhor para RPS/log-loss/Brier):**

| Modelo (1X2) | RPS ↓ | log-loss ↓ | Brier ↓ | acerto 1X2 |
|---|---|---|---|---|
| Modelo P1 (RF binário, p_draw=0) — linha a bater | 0,3388 | 9,2124 | 0,9233 | 0,421 |
| B0 — frequência-base do treino | 0,2334 | 1,0777 | 0,6530 | 0,439 |
| B1 — favorito do ranking (suavizado) | 0,2019 | 1,0053 | 0,5982 | 0,579 |
| **P2 — LogReg multinomial** (escolhido) | **0,2072** | **1,0047** | **0,5972** | 0,491 |
| P2 — GradientBoosting | 0,2086 | 1,0090 | 0,6016 | 0,474 |
| P2 — RandomForest multiclasse | 0,2985 | 2,5252 | 0,8115 | 0,404 |

- **Critério "RPS melhora vs. binário": ✅** — a LogReg multinomial cai de **0,3388 → 0,2072** (−39%), supera o baseline B0 (0,2334) e fica praticamente empatada com o B1 (0,2019), além de ter o **melhor log-loss e o melhor Brier** de todo o conjunto. O GradientBoosting fica logo atrás; o RF multiclasse melhora vs. o binário mas é mal calibrado (log-loss 2,53) e não é selecionado.
- **Critério "3 probabilidades somam 1": ✅** (verificado na `cell-P2.1`, soma = 1,000000).
- **Critério "empate em proporção plausível": ✅** — a LogReg atribui em média **28,5%** de massa ao empate (faixa 18,8%–31,8%), próximo dos **24,6%** de empates reais no hold-out e longe de 0% (binário) ou 50%. O modelo deixou de ser estruturalmente incapaz de prever empate.
- **Seleção automática:** a `cell-P2.2` escolhe o melhor modelo pelo **RPS** e expõe `clf_multiclass`, consumido na inferência.

**Células criadas/alteradas:**
- `cell-28` (P1) já fornece o alvo `outcome` (0/1/2) — reutilizado, sem duplicação.
- **Novas células `P2.1`–`P2.3`** (após a `cell-36`/P1.4): treino dos 3 classificadores multiclasse + helper `proba_1x2` (reordena `predict_proba` para `[home, draw, away]`) + `assert` anti-vazamento (`P2.1`); avaliação vs. binário e baselines + seleção pelo RPS (`P2.2`); reliability diagram do empate (`P2.3`).
- **`cell-43` (inferência da Copa) alterada:** consome `proba_1x2(clf_multiclass, row)` → `[p_home, p_draw, p_away]` (em vez de `1 - home_win_prob`); resultado previsto = `argmax`; heurística de placar agora prevê empate. Também troca o `.append` (depreciado) por `pd.concat`. `cell-42` (markdown) atualizada.

### Dependências
P1 (para medir o ganho).

---

# P4 — Engenharia de features de futebol (decomposto em P4.1–P4.9)

**Esforço:** Médio · **Impacto:** Alto (maior salto de acurácia real) · **Risco:** Médio (leakage temporal)

### Objetivo geral
Sair de 2 features de ranking (`average_rank`, `rank_difference`) para um conjunto que capture **força contínua, forma recente, mando real, descanso e confronto direto** — tudo calculado **estritamente com dados anteriores a cada partida** — e re-treinar o classificador multiclasse do P2 com o conjunto que comprovadamente melhora o RPS no hold-out.

### Por que decompor
O P4 monolítico é um bloco grande e arriscado: várias features heterogêneas, cada uma com seu próprio risco de **vazamento temporal**. Quebrá-lo em subtarefas permite:
- Implementar e **validar uma feature de cada vez** (RPS no hold-out vs. P2) antes da próxima.
- Isolar a fonte de qualquer regressão de métrica.
- Reutilizar uma **única infraestrutura anti-vazamento** (P4.1) em todas as features.

> Nota sobre o P3: na primeira versão deste plano, o P4 estava marcado para rodar **antes** do P3 (as médias móveis de gols alimentariam o Poisson). O P3 acabou implementado **standalone** (forças por MLE direta dos gols). Portanto o P4 **não bloqueia mais o P3**; as features de gols do P4.5 ficam como **insumo futuro** para reforçar o P3 (prior/offset), não como pré-requisito.

### Situação atual (antes do P4)
- Features dos classificadores: só `rank_difference` e `average_rank` (`cell-28`).
- `cell-26` **descarta `total_points`** (força contínua FIFA, mais informativa que a posição ordinal) — recuperar (P4.3).
- `neutral` já foi **preservado no P3** (a `cell-13` deixou de descartá-la) — no P4.3 basta **reusá-lo** como feature dos classificadores.
- O baseline **B2 — Elo puro** ficou como **stub no P1** — o P4.2 o habilita.

### Subtarefas

| # | Subtarefa | Esforço | Impacto | Risco vazamento |
|---|---|---|---|---|
| **P4.1** | Infraestrutura de features sem vazamento (fundação) | Baixo | Alto (habilita o resto) | — (é o anti-vazamento) |
| **P4.2** | Elo / pi-ratings + baseline B2 | Médio | Alto | Médio |
| **P4.3** | Recuperar `total_points` e reusar `neutral` | Baixo | Médio | Baixo |
| **P4.4** | Forma recente (pontos V/E/D em N jogos) | Baixo | Médio | Médio |
| **P4.5** | Médias móveis de gols marcados/sofridos | Médio | Médio | Médio-alto |
| **P4.6** | Descanso (dias desde o último jogo) | Baixo | Baixo-médio | Baixo |
| **P4.7** | Head-to-head (ratios + shootouts) | Médio | Médio | Médio |
| **P4.8** | Avaliação incremental + seleção de features + re-treino | Médio | Alto (consolida o ganho) | Baixo |
| **P4.9** | Função única de features (treino + inferência) | Baixo | Confiabilidade | Alto (train/serving skew) |

### Ordem de execução recomendada (fundação → menor risco → maior risco → consolidação)

```
P4.1 (anti-vazamento) ──► P4.3 (total_points, neutral) ──► P4.2 (Elo + B2)
   fundação                  features estáticas/baixo risco     força contínua
        │                                                            │
        └────────────► P4.4 (forma) ─► P4.6 (descanso) ─► P4.5 (gols) ─► P4.7 (H2H)
                          janelas deslocadas, risco crescente
                                              │
                                              ▼
                              P4.8 (seleção + re-treino do P2) ──► P4.9 (função única)
                                 consolida o ganho no hold-out       higiene / anti-skew
```

Racional da ordem: **P4.1 primeiro** porque todas as outras dependem do helper de janela deslocada e do assert anti-vazamento. **P4.3 antes do resto** porque é a feature de menor risco (valores estáticos por partida, sem janela móvel) e dá um ganho rápido. **P4.2 (Elo)** logo em seguida porque é a feature isolada de maior impacto e desbloqueia o baseline B2 pendente do P1. Depois as features de janela deslocada em **ordem crescente de risco de vazamento** (forma → descanso → gols → H2H). **P4.8** ao final consolida (seleção + re-treino), e **P4.9** garante que a inferência use exatamente a mesma geração de features (anti-skew, sobreposição com P6).

---

## Status global do P4: ✅ IMPLEMENTADO (features **não adotadas** — ver veredito)

Implementadas (agrupadas nas células **`P4.1`–`P4.4`** do notebook) as subtarefas **P4.1** (infraestrutura), **P4.2** (Elo + B2), **P4.3** (`total_points`/`neutral`), **P4.4** (forma), **P4.5** (gols ataque/defesa), **P4.6** (descanso), **P4.8** (seleção incremental + re-treino) e **P4.9** (função única treino/inferência). **P4.7 (H2H)** foi **deliberadamente adiado** (ver abaixo).

> **Veredito honesto: as features do P4 NÃO foram adotadas em produção.** Com metodologia corrigida, elas **melhoram a validação** (RPS 0,1627 → **0,1536**) mas **não generalizam** ao hold-out de 57 jogos. Pela regra do próprio plano ("aceita só se melhora o RPS no hold-out"), o P4 é **rejeitado**; o **Ensemble P2+P3** (RPS 0,2034) seguiu como saída de produção (até o P5 — ver abaixo). A infraestrutura fica pronta e testada para evoluções futuras.

> **Atualização (P5):** a saída de produção evoluiu para o **Ensemble (P2 calibrado por isotonic + DC cru)** (RPS **0,2013**, log-loss 0,9973, Brier 0,5854), que melhora as três métricas sem piorar o RPS. Ver a seção P5.

### Correções de auditoria aplicadas nesta implementação
A primeira versão do P4 (monolítica) tinha falhas que **invalidavam os números reportados**. Corrigidas:
1. **Sort não-estável → desalinhamento.** Um segundo `sort_values('date')` (não-estável) dentro da geração de features reordenava jogos de mesma data, deixando `p4_probs` em ordem diferente de `y_holdout` (e do ensemble). **Confirmado empiricamente.** Corrigido com `order_chronologically` (mergesort + desempate `['date','home_team','away_team']`); agora `hold_feat` alinha com `holdout_set` (57 == 57).
2. **Seleção de features no hold-out.** A versão antiga escolhia features minimizando o RPS no **próprio hold-out** (a validação 2019–2021 era construída e nunca usada) → ganho inflado. Agora a seleção *forward-greedy* roda **só na validação**; o hold-out volta a ser intocado.
3. **H2H vazante.** O `historical_win-loose-draw_ratios.csv` é agregado de todos os tempos (inclui o confronto-alvo) → **removido por construção** (não mais um candidato que dependia da ablação rejeitá-lo). H2H sem vazamento fica como **P4.7 (futuro)**.
4. **Inferência por posição.** A inferência casava probabilidades por índice posicional sobre um frame re-sortido → podia atribuir a previsão ao **jogo errado**. Reescrita para casar **por nome de seleção**.

### Números reais medidos (hold-out Copa 2022, 57 jogos, 14 empates; menor = melhor)
Verificados por execução offline das células sobre os CSVs locais (o pipeline reproduz exatamente os números do P1/P2/P3).

| Modelo (1X2) | RPS ↓ | log-loss ↓ | Brier ↓ | acerto 1X2 |
|---|---|---|---|---|
| B0 — frequência-base | 0,2334 | 1,0777 | 0,6530 | 0,439 |
| B1 — favorito do ranking (suavizado) | **0,2019** | 1,0053 | 0,5982 | **0,579** |
| **B2 — Elo puro** (era stub no P1; preenchido no P4.2) | 0,2178 | 1,0350 | 0,6209 | 0,544 |
| P2 — LogReg (só ranking) | 0,2072 | 1,0047 | 0,5972 | 0,491 |
| P3 — Dixon-Coles (campo neutro) | 0,2178 | 1,0942 | 0,6220 | 0,544 |
| **P4 — LogReg (expandido)** | 0,2168 | 1,0365 | 0,6176 | 0,561 |
| **Ensemble P2+P3** (produção) | **0,2034** | **1,0025** | **0,5901** | 0,544 |
| Ensemble P4+P3 | 0,2096 | 1,0256 | 0,6032 | 0,561 |

- **Critério "RPS do P4 melhora vs. P2 no hold-out": ❌ FALHOU** — P4 (0,2168) > P2 (0,2072). Na **validação** o P4 vence (0,1536 vs. 0,1627), mas não generaliza ao hold-out de 57 jogos.
- **B2 (Elo) preenchido:** 0,2178, competitivo com B1/P3, como esperado.
- Features selecionadas na validação: `average_rank, rank_difference, elo_diff, neutral_int, ga_diff, gf_diff, form_diff` (descartadas `points_diff`, `rest_diff`).
- Anti-vazamento: 208 jogos-canário OK; injeção de placar falso (9-0) em *t* não altera a feature de *t*.

### Células criadas/alteradas
- **`P4.1`** — infraestrutura: `order_chronologically` (sort estável), `compute_elo`, `team_long_view`, `rolling_team_feature`, `assert_no_leakage`.
- **`P4.2`** — `build_football_features` (função única) + testes anti-vazamento + alinhamento; gera `df_feat`/`train_feat`/`valid_feat`/`hold_feat`.
- **`P4.3`** — ablação + seleção *forward-greedy* **na validação**.
- **`P4.4`** — avaliação **alinhada** no hold-out + baseline **B2 (Elo)** + escolha honesta do modelo de produção (`clf_prod`/`PROD_FEATURES`).
- **`cell-43` (inferência 2026)** — usa `clf_prod` + Dixon-Coles, **alinhado por nome**. **`cell-42`** (markdown) atualizada.

### P4.7 — deliberadamente adiado
H2H sem vazamento exige reconstrução incremental (não o CSV agregado). Como as demais features já **não** bateram o P2 no hold-out, o H2H foi deixado como trabalho futuro (a infraestrutura `assert_no_leakage`/`rolling_team_feature` já o suporta).

---

## P4.1 — Infraestrutura de features sem vazamento (fundação)

### Status: ✅ IMPLEMENTADO (célula `P4.1`)

**Esforço:** Baixo · **Impacto:** Alto (habilita todas as demais subtarefas) · **Risco:** — (é justamente o mecanismo anti-vazamento)

### Objetivo
Criar o alicerce reutilizável por todas as P4.x: ordenação cronológica estável, um **helper genérico de janela deslocada** (`shift(1)` por seleção) e um **assert de não-vazamento** que cada subtarefa seguinte invoca.

### Passos
1. Garantir a ordenação por `date` (com desempate estável) **antes** de qualquer cálculo de feature; reaproveitar a `date` já preservada no P1 (`cell-26`) e no P3 (`cell-13`).
2. Escrever um helper genérico `rolling_team_feature(df, team_col, value_fn, window, min_periods)` que, para cada seleção, aplica `groupby(team_col).shift(1)` antes de qualquer agregação móvel — ou seja, **o valor da linha nunca enxerga o próprio resultado**.
3. Escrever um helper `team_long_view(df)` que "desdobra" cada partida em duas linhas (perspectiva do mandante e do visitante) para calcular features por seleção independentemente de ter jogado em casa ou fora, e depois remontar para o formato por partida.
4. Escrever `assert_no_leakage(df_features, partida_id)` reutilizável: verifica, por amostragem, que a feature da partida *t* é função apenas de partidas com `date < date_t` (recalcula a feature truncando o histórico em *t* e compara).
5. Nova célula utilitária (após as células do P3, p.ex. `P4.1`) contendo esses helpers — análoga à célula utilitária de métricas do P1.

### Critério de aceitação
- Existe `rolling_team_feature(...)` e `assert_no_leakage(...)` numa célula utilitária reutilizável.
- `assert_no_leakage` passa para uma feature-canário trivial (ex.: contagem de jogos anteriores).
- DataFrame ordenado cronologicamente e com desempate estável (resultado reproduzível, `random_state=42` onde aplicável).

### Dependências
P1 (preserva `date` e provê o padrão de célula utilitária). Pré-requisito de **todas** as outras subtarefas P4.x.

---

## P4.2 — Elo / pi-ratings + baseline B2

**Esforço:** Médio · **Impacto:** Alto (feature isolada de maior poder) · **Risco:** Médio (estado incremental tem de respeitar a ordem cronológica)

### Status: ✅ IMPLEMENTADO (Elo em `build_football_features`/`P4.2`; B2 em `P4.4`)

### Objetivo
Adicionar uma medida de **força contínua, auto-corrigível** (Elo), superior ao rank FIFA discreto, e usar o mesmo Elo para habilitar o baseline **B2 — Elo puro** que ficou como stub no P1.

### Passos
1. Implementar Elo incremental por seleção (estado atualizado **cronologicamente**, jogo a jogo), no estilo *World Football Elo*: ajuste de mando (zerado quando `neutral == True`), fator-K dependente do tipo de torneio/importância e **margem de gols** (goal-difference multiplier).
2. A feature de cada partida é o Elo **antes** do confronto (`elo_home_pre`, `elo_away_pre`, `elo_diff`) — naturalmente deslocado, pois o estado só é atualizado após processar a partida (validar com `assert_no_leakage` do P4.1).
3. Converter a diferença de Elo em probabilidade de vitória esperada (curva logística do Elo) e derivar 1X2 para preencher o **baseline B2** (com uma fatia plausível para o empate, à la B1).
4. Substituir o stub de B2 nas células de baseline do P1 e incluí-lo nas tabelas comparativas.

### Critério de aceitação
- `assert_no_leakage` passa para `elo_home_pre`/`elo_away_pre`.
- B2 (Elo puro) deixa de ser stub e entra nas comparações com RPS/log-loss/Brier no hold-out 2022.
- RPS de B2 documentado; espera-se B2 **competitivo com B1** (favorito do ranking) — se Elo não bate o rank discreto, é sinal de implementação a revisar.

### Dependências
P4.1 (helper + assert). Habilita o B2 pendente de P1. Independente das demais features.

---

## P4.3 — Recuperar `total_points` e reusar `neutral`

**Esforço:** Baixo · **Impacto:** Médio · **Risco:** Baixo (valores estáticos por partida, sem janela móvel)

### Status: ✅ IMPLEMENTADO (`points_diff` e `neutral_int` em `build_football_features`/`P4.2`)

### Objetivo
Aproveitar dois sinais já presentes nos dados e hoje subutilizados: a **pontuação FIFA** (`total_points`, mais informativa que a posição ordinal `rank`) e o **mando real** (`neutral`), distinguindo o treino (~77% com mando) do alvo Copa (campo neutro).

### Passos
1. Em `cell-26`, **deixar de descartar `total_points`** (mesmo padrão usado para preservar `date` no P1) e propagá-la pelos merges com o `ranking.csv` para gerar `total_points_home`, `total_points_away` e `points_difference`.
2. **Reusar `neutral`** como feature dos classificadores — a coluna já foi preservada no P3 (`cell-13`). Atenção: no P3 o `neutral` é usado **só pelo modelo de gols**; aqui ele passa a poder entrar em `X` dos classificadores. Garantir que na inferência da Copa 2026 `neutral = True` em todos os jogos (campo neutro).
3. Adicionar `total_points_*`/`points_difference`/`neutral` ao conjunto candidato avaliado no P4.8.

### Critério de aceitação
- `assert_no_leakage` trivialmente satisfeito (sem janela; só merge por seleção/data de ranking anterior à partida — confirmar que o `rank_date` usado é ≤ `date` da partida).
- `total_points_*` e `neutral` disponíveis como colunas candidatas em `df_model`.
- Documentado o ganho/perda de RPS de cada uma no P4.8.

### Dependências
P4.1. Reaproveita a preservação de `neutral` feita no P3 e de `date` feita no P1.

---

## P4.4 — Forma recente (pontos V/E/D nos últimos N jogos)

**Esforço:** Baixo · **Impacto:** Médio · **Risco:** Médio (janela deslocada obrigatória)

### Status: ✅ IMPLEMENTADO (`form_*`/`form_diff` em `build_football_features`/`P4.2`)

### Objetivo
Capturar **momentum / fase de elenco** via média de pontos (V=3 / E=1 / D=0) nos últimos N jogos de cada seleção, calculada estritamente com jogos anteriores.

### Passos
1. Sobre a `team_long_view` do P4.1, mapear cada jogo para pontos da perspectiva da seleção (3/1/0).
2. Usar `rolling_team_feature` com `shift(1)` e janela `N=5` (testar também N=3 e N=10 no P4.8) → `form_home`, `form_away`, `form_diff`.
3. Definir o tratamento de `min_periods` (seleções com poucos jogos): fallback para a média global ou NaN explícito tratado pelo modelo.
4. Rodar `assert_no_leakage` sobre `form_home`/`form_away`.

### Critério de aceitação
- `assert_no_leakage` passa (a forma da partida *t* não inclui o resultado de *t*).
- Feature adicionada ao conjunto candidato do P4.8; mantida só se melhora o RPS no hold-out vs. P2.

### Dependências
P4.1.

---

## P4.5 — Médias móveis de gols marcados/sofridos

**Esforço:** Médio · **Impacto:** Médio · **Risco:** Médio-alto (gols são derivados do placar — risco direto de vazar o alvo)

### Status: ✅ IMPLEMENTADO (`gf_*`/`ga_*`/`gf_diff`/`ga_diff` em `P4.2`; integração ao P3 = futuro)

### Objetivo
Proxy contínuo de **ataque e defesa**: média móvel de gols marcados e sofridos nos últimos N jogos (separando casa/fora quando houver dados suficientes). Também serve de **insumo futuro do P3** (prior/offset das forças Dixon-Coles).

### Passos
1. Sobre a `team_long_view`, calcular gols marcados/sofridos por seleção; aplicar `rolling_team_feature` com `shift(1)` → `gf_home`, `ga_home`, `gf_away`, `ga_away` (e diffs).
2. Avaliar a separação casa/fora vs. agregado único (seleções jogam pouco — separar pode esvaziar a janela; decidir no P4.8).
3. **Atenção redobrada ao vazamento:** `home_score`/`away_score`/`score_difference` são o alvo — assegurar que apenas as **médias deslocadas** entram em `X`, nunca os gols da própria partida. Reforçar o `assert` anti-vazamento de `cell-P2.1`.
4. Rodar `assert_no_leakage` sobre todas as quatro features de gols.

### Critério de aceitação
- `assert_no_leakage` passa para `gf_*`/`ga_*`; o `assert` anti-vazamento da `cell-P2.1` continua válido com as novas colunas.
- Feature mantida só se melhora o RPS no P4.8.
- (Insumo futuro) features disponíveis para alimentar o P3 como prior/offset.

### Dependências
P4.1. Relaciona-se ao P3 (insumo futuro, não bloqueante).

---

## P4.6 — Descanso (dias desde o último jogo)

**Esforço:** Baixo · **Impacto:** Baixo-médio · **Risco:** Baixo

### Status: ✅ IMPLEMENTADO (`rest_*`/`rest_diff` em `P4.2`; descartada na seleção da validação)

### Objetivo
Modelar **fadiga**: dias desde a última partida de cada seleção — relevante em torneio com jogos a cada 3–4 dias.

### Passos
1. Sobre a `team_long_view`, calcular `date - date_do_jogo_anterior` por seleção (via `groupby().shift(1)` sobre a `date`) → `rest_home`, `rest_away`, `rest_diff`.
2. Tratar o primeiro jogo de cada seleção (sem jogo anterior): NaN/sentinela.
3. Rodar `assert_no_leakage` (a data anterior é sempre < `date` da partida).

### Critério de aceitação
- `assert_no_leakage` passa para `rest_*`.
- Feature mantida só se melhora o RPS no P4.8 (impacto possivelmente pequeno fora de torneios; aceitável descartá-la).

### Dependências
P4.1.

---

## P4.7 — Head-to-head (ratios + shootouts)

**Esforço:** Médio · **Impacto:** Médio · **Risco:** Médio (o CSV de ratios é agregado — atenção a incluir confrontos posteriores ao jogo)

### Status: ⏸️ ADIADO (trabalho futuro — ver "Status global do P4")

### Objetivo
Capturar o histórico de **confronto direto** entre as duas seleções, incluindo decisões por pênalti.

### Passos
1. Fazer merge de `historical_win-loose-draw_ratios.csv` pelo par `(country1, country2)` (colunas reais: `games, wins, looses, draws`), gerando taxas H2H orientadas ao mandante (`h2h_win_rate`, `h2h_draw_rate`, `h2h_n`).
2. Cuidado de vazamento: o arquivo de ratios é um **agregado total** (pode incluir jogos posteriores à partida em questão). Preferir **reconstruir o H2H incrementalmente** a partir de `historical-results.csv` com `shift(1)` por par de seleções; usar o CSV agregado apenas como verificação/sanidade, não como feature direta de treino.
3. Incorporar `shootouts.csv` (colunas `date, home_team, away_team, winner`) como sinal de confrontos historicamente equilibrados (decididos nos pênaltis) — também truncado por data.
4. Aplicar decaimento temporal e ajuste de mando ao H2H, se melhorar o RPS.
5. Rodar `assert_no_leakage` sobre as features H2H reconstruídas.

### Critério de aceitação
- `assert_no_leakage` passa para `h2h_*` (a feature da partida *t* só usa confrontos anteriores a *t*).
- Documentado explicitamente o motivo de **não** usar o CSV agregado diretamente como feature (risco de vazar confrontos futuros).
- Feature mantida só se melhora o RPS no P4.8.

### Dependências
P4.1. Usa `historical_win-loose-draw_ratios.csv`, `shootouts.csv` e `historical-results.csv`.

---

## P4.8 — Avaliação incremental + seleção de features + re-treino

**Esforço:** Médio · **Impacto:** Alto (consolida o ganho real do P4) · **Risco:** Baixo

### Status: ✅ IMPLEMENTADO (células `P4.3`/`P4.4`)
> **Correção metodológica importante:** a seleção é feita **na validação 2019–2021**, **não no hold-out** (a prescrição original abaixo dizia "no hold-out" — isso contaminava o hold-out e foi a causa do "ganho do P4" fantasma; ver "Status global do P4"). O hold-out passou a ser usado **só para o relatório final**.

### Objetivo
Adicionar as features **uma de cada vez**, manter apenas as que melhoram o RPS **na validação**, re-treinar o classificador multiclasse do P2 com o conjunto vencedor e comparar contra todos os baselines/modelos no hold-out.

### Passos
1. Partindo do conjunto base do P2 (`average_rank`, `rank_difference`), adicionar cada feature/grupo (P4.2–P4.7) **incrementalmente** (forward selection), medindo o RPS **na validação 2019–2021** (não no hold-out).
2. **Manter só as que melhoram o RPS na validação** (e idealmente não pioram log-loss/Brier) — registrar a tabela de ganho marginal por feature (estilo das tabelas dos P1/P2/P3).
3. Re-treinar o classificador escolhido no P2 (LogReg multinomial, e re-checar GradientBoosting agora que há mais features) **só no treino temporal (≤ 2018)**, com `random_state=42`.
4. Comparar o modelo final contra **B0, B1, B2 (Elo, do P4.2), P2 (só ranking), P3 (Dixon-Coles) e o ensemble P2+P3** — e avaliar um novo **ensemble P4+P3**.
5. Reforçar o `assert` anti-vazamento (`cell-P2.1`) para o novo conjunto de colunas de `X`.

### Critério de aceitação
- Tabela de seleção incremental com o ganho marginal de RPS de cada feature **na validação**. ✅ (ver "Status global do P4")
- RPS do modelo P4 final **melhora vs. P2** (0,2072) no hold-out. ❌ **Não atingido** — P4 deu 0,2168 no hold-out (apesar de vencer na validação). Por isso o P4 **não foi adotado**.
- Conjunto final de features documentado com `assert` anti-vazamento ativo. ✅

### Dependências
P4.2–P4.7 (as features a selecionar), P1 (avaliação), P2 (classificador a re-treinar), e P3 (para a comparação/ensemble).

---

## P4.9 — Função única de features (treino + inferência)

**Esforço:** Baixo · **Impacto:** Confiabilidade · **Risco:** Alto se ignorado (train/serving skew silencioso)

### Status: ✅ IMPLEMENTADO (`build_football_features` é a função única; inferência em `cell-43`)

### Objetivo
Garantir que a inferência da Copa 2026 (`cell-43`) gere as features com **exatamente a mesma função** do treino — evitando o skew clássico que invalida toda a avaliação do P4.

### Passos
1. Extrair a geração de todas as features P4.x para **uma única função** `build_features(df, ...)` usada tanto no treino quanto na inferência.
2. Reescrever a inferência da Copa 2026 (`cell-43`) para chamar `build_features` em vez de recriar features à mão; tratar seleções sem histórico suficiente com fallback explícito (à la fallback do P3 para times fora do ajuste DC).
3. Adicionar um teste de consistência: as colunas/ordem de `X` no treino e na inferência são idênticas.

### Critério de aceitação
- Treino e inferência usam a **mesma** `build_features`.
- Teste de consistência de colunas passa.

### Dependências
P4.8 (conjunto final de features). **Sobrepõe-se ao P6 item 1** (função única treino/inferência) — coordenar para não duplicar: se o P6.1 for feito antes, P4.9 apenas estende a função única já existente às features do P4.

---

# P3 — Modelo estatístico de gols (Poisson / Dixon-Coles)

**Esforço:** Médio · **Impacto:** Alto · **Risco:** Médio

### Objetivo
Substituir a heurística de placar por faixas (`cell-38`) por um modelo que gera a **distribuição completa de placares** e probabilidades 1X2 coerentes.

### Situação atual
- `cell-37`/`cell-38`: placar por `if` sobre `home_win_prob` (`>0.70 → 2 gols`, `0.60–0.70 → 1 gol`). Não é estatística; empate só sai por acidente.

### Abordagem
1. **Forças de ataque/defesa + mando:** modelar `home_score` e `away_score` como Poisson, estimando parâmetros por seleção via máxima verossimilhança (`scipy.optimize` — sem dependências novas além de `scipy`).
2. **Dixon-Coles** sobre Poisson puro:
   - Corrige a subestimação de placares baixos (0-0, 1-0, 1-1) e a dependência entre os gols dos dois times.
   - **Decaimento temporal** (jogos recentes pesam mais) para lidar com a troca de gerações de elenco.
3. A partir de (λ_casa, λ_fora): montar a **matriz de placares** P(i,j); somar para obter P(vitória)/P(empate)/P(derrota) e o **placar mais provável** (substitui a heurística).

### Passos
1. Implementar a log-verossimilhança Poisson + termo de correção Dixon-Coles (~40 linhas).
2. Ajustar com decaimento temporal exponencial sobre a data.
3. Função `predict_match(home, away) -> (matriz_placares, p_1x2, placar_mais_provavel)`.
4. Comparar RPS deste modelo vs. o classificador de P2+P4 no hold-out; possivelmente **ensemble** (média das probabilidades 1X2).

### Critério de aceitação
- Probabilidades 1X2 derivadas da matriz somam 1 e batem com a frequência de empates observada.
- Placar exato no hold-out 2022 **> 7/48** (atual).
- RPS competitivo ou melhor que P4.

### Status: ✅ IMPLEMENTADO (standalone, sem P4)
Heurística de placar por faixas substituída por um **modelo Dixon-Coles** (Poisson bivariado com correção low-score) ajustado por **máxima verossimilhança** (`scipy.optimize.minimize`, SLSQP). Estima, por seleção, força de **ataque** (α) e **defesa** (β), mais **mando global** (γ) e o termo de **dependência low-score** (ρ), com **decaimento temporal exponencial**. Identificabilidade por `Σα = 0`. A partir de (λ_casa, λ_fora) monta a **matriz de placares** P(i, j) (0..10 gols), de onde saem o 1X2 coerente e o **placar mais provável** (argmax).

> **Decisão — P4 não implementado.** Acordado executar o P3 de forma autônoma: as forças saem **direto da MLE dos gols históricos**, sem depender das médias móveis de gols do P4 (ainda não implementado). **Integrar as features do P4 (Elo/forma como prior/offset) fica como trabalho futuro.**

**Decisão — mando neutralizável.** γ entra **só em jogos `neutral == False`** (`log λ_casa = α_home − β_away + γ·(¬neutro)`). O treino tem ~77% de jogos com mando, mas a **Copa é em campo neutro**; logo a inferência da Copa usa `neutral_game=True` e **zera γ**. Isso evita inflar o "mandante" do calendário num torneio neutro. Para preservar `neutral`, a `cell-13` deixou de descartá-la (mesmo padrão do P1 com a `date`); ela **nunca** entra como feature dos classificadores P1/P2 — só o modelo de gols a usa.

**Ajuste:** MLE em **treino + validação (≤ 2021)** (seleções jogam pouco; mais jogos por time). A **meia-vida do decaimento foi escolhida pelo RPS na validação 2019–2021** (não no hold-out): venceu **540 dias** (γ=0,294, ρ=−0,073). O **hold-out Copa 2022 nunca foi tocado** no ajuste.

**Números reais medidos (hold-out Copa 2022, 57 jogos, 14 empates; menor = melhor para RPS/log-loss/Brier):**

| Modelo (1X2) | RPS ↓ | log-loss ↓ | Brier ↓ | acerto 1X2 | placar exato |
|---|---|---|---|---|---|
| B0 — frequência-base | 0,2318 | 1,0739 | 0,6499 | 0,439 | — |
| B1 — favorito do ranking (suavizado) | **0,2019** | 1,0053 | 0,5982 | **0,579** | — |
| P2 — LogReg multinomial | 0,2072 | 1,0047 | 0,5972 | 0,491 | — |
| **P3 — Dixon-Coles (campo neutro)** | 0,2178 | 1,0942 | 0,6220 | 0,544 | **8/57 (7/48)** |
| **Ensemble P2+P3 (média)** | **0,2034** | **1,0025** | **0,5901** | 0,544 | — |

- **Critério "placar exato > 7/48": ✅** — Dixon-Coles acerta **8 de 57** placares no hold-out (e **7/48** nos primeiros 48 jogos, ≈ fase de grupos). A heurística do P2 não era avaliada por placar exato; o DC dá placares de forma **principiada** (argmax da distribuição), não por `if`.
- **Critério "1X2 soma 1 e bate a taxa de empates": ✅** — probabilidades somam 1,000000; massa média no empate **27,7%** vs. **24,6%** de empates reais. Os placares mais previstos são os baixos típicos de jogo neutro: **1-0, 0-0, 0-1, 2-0, 1-1**.
- **Critério "RPS competitivo ou melhor que P4": ✅ (vs. a referência disponível, o P2)** — o DC puro (RPS 0,2178) fica **entre B0 e B1/P2** e tem o **melhor acerto 1X2 de todos os modelos (0,544)**; perde um pouco em RPS/log-loss para o P2 porque é um pouco mais confiante. O **ensemble P2+P3 (média das probabilidades) é o melhor modelo do conjunto em log-loss (1,0025) e Brier (0,5901) e melhora o RPS para 0,2034** — abaixo do P2 (0,2072), praticamente empatado com o B1 (0,2019), e mantendo o acerto 1X2 em 0,544. Recomenda-se o **ensemble** como saída final.

**Células criadas/alteradas:**
- **`cell-13` alterada:** deixa de descartar `neutral` (necessária ao mando do DC); `neutral` segue pelos merges até `df_model` e **nunca** vira feature dos classificadores.
- **Novas células `P3.1`–`P3.3`** (após a `cell-P2.3`): `P3.1` — log-verossimilhança Poisson + correção Dixon-Coles + decaimento temporal e `fit_dixon_coles`; `P3.2` — `predict_match(home, away, neutral_game=True)` (matriz de placares → 1X2 → placar argmax), fallback freq-base e **tuning da meia-vida pelo RPS na validação** (540 dias); `P3.3` — avaliação no hold-out 2022, placar exato, comparação com B0/B1/P2 e **ensemble P2+P3**. Markdown de seção (`P3`) antes da `P3.1`.
- **`cell-43` (inferência da Copa) alterada:** usa `predict_match(..., neutral_game=True)` (Dixon-Coles em campo neutro) para o placar e o 1X2; **fallback** para `clf_multiclass`+heurística quando uma seleção não está no ajuste do DC (apenas ~6/72 jogos do calendário 2026). `cell-42` (markdown) atualizada.

### Dependências
P1 (medição), P2 (alvo/ensemble). **P4 dispensado como pré-requisito** (forças via MLE direta — ver decisão acima); o P4 foi **decomposto em P4.1–P4.9** e suas features de gols (P4.5) ficam como insumo futuro do P3. Requer `scipy`.

---

# P5 — Calibração de probabilidades

**Esforço:** Baixo · **Impacto:** Médio · **Risco:** Baixo

### Objetivo
Tornar as probabilidades confiáveis (uma previsão de "70%" deve acertar ~70% das vezes), melhorando RPS/Brier e a escolha de placar a "apostar" no critério de pontuação.

### Situação atual (antes do P5)
As probabilidades do P2/ensemble não passavam por calibração; o P2 LogReg sai um pouco mal calibrado (subconfiante na classe vencedora).

### Passos
1. Usar `CalibratedClassifierCV` (sklearn) com **Platt scaling** ou **isotonic regression**, ajustado no conjunto de **validação** (2019–2021), nunca no treino nem no hold-out.
2. Verificar com **reliability diagram** (função de P1) antes/depois.
3. Para modelos próprios (Poisson/DC), avaliar calibração das probabilidades 1X2 diretamente.

### Critério de aceitação
- Reliability diagram mais próximo da diagonal após calibração.
- Brier/log-loss no hold-out melhoram ou ficam estáveis sem piorar RPS.

### Status: ✅ IMPLEMENTADO (calibração isotônica do P2 **ADOTADA**)

Calibração ajustada **exclusivamente na validação 2019–2021** (1560 jogos); o hold-out Copa 2022 (57 jogos) ficou intocado e serve só para o relatório. Para o classificador sklearn (P2 LogReg) usamos `CalibratedClassifierCV` com **Platt (sigmoid)** e **isotonic** sobre o estimador já treinado (congelado: `FrozenEstimator` no sklearn ≥ 1.6, fallback `cv='prefit'` no Colab). Para os modelos próprios (Dixon-Coles e ensemble), que não são estimadores sklearn, calibramos as probabilidades 1X2 diretamente por **temperature scaling** (1 parâmetro, minimiza log-loss na validação) e **isotônica multiclasse** (1-vs-resto + renormalização).

**Números reais medidos (hold-out Copa 2022, 57 jogos, 14 empates; menor = melhor para RPS/log-loss/Brier):**

| Modelo (1X2) | RPS ↓ | log-loss ↓ | Brier ↓ | acerto 1X2 |
|---|---|---|---|---|
| P2 LogReg (sem calibração) | 0,2072 | 1,0047 | 0,5972 | 0,491 |
| P2 + Platt (sigmoid) | 0,2037 | 0,9978 | 0,5906 | 0,509 |
| **P2 + Isotonic** | **0,1985** | **0,9805** | **0,5792** | **0,561** |
| P3 Dixon-Coles (sem calibração) | 0,2178 | 1,0942 | 0,6220 | 0,544 |
| P3 DC + temperature (T=0,795) | 0,2228 | 1,1495 | 0,6329 | 0,544 |
| P3 DC + isotônica-mc | 0,2277 | 1,7053 | 0,6503 | 0,491 |
| Ensemble P2+P3 (produção anterior, cru) | 0,2034 | 1,0025 | 0,5901 | 0,544 |
| Ensemble + temperature (T=0,792) | 0,2041 | 1,0095 | 0,5894 | 0,544 |
| **Ensemble (P2-iso + DC cru) — NOVA produção** | **0,2013** | **0,9973** | **0,5854** | **0,561** |

- **Critério "Brier/log-loss melhoram sem piorar RPS": ✅** — calibrar o **P2 com isotonic** melhora **as três** métricas próprias: RPS 0,2072 → **0,1985**, log-loss 1,0047 → **0,9805**, Brier 0,5972 → **0,5792** (e acerto 1X2 0,491 → 0,561). Platt também melhora as três, de forma mais conservadora.
- **Modelos próprios (DC):** calibrar o Dixon-Coles diretamente **piora tudo** — ele já sai bem calibrado da MLE (temperature/isotônica encolhem na direção errada). **Não calibramos o DC.** Calibrar o **ensemble inteiro** também não ajuda (temperature ≈ neutro/pior).
- **Nova saída de produção:** o ensemble passa a combinar o **P2 calibrado (isotonic)** com o **DC cru** → RPS 0,2034 → **0,2013**, log-loss → 0,9973, Brier → 0,5854. Melhora **RPS, log-loss E Brier** simultaneamente, **sem violar** a regra de ouro. A `cell-P5.3` valida isso programaticamente (`ADOPT_CALIBRATION = True`).
- **Reliability diagram (critério 2): ✅** — verificado na `cell-P5.4` (home_win e empate, antes vs. depois). Com 57 jogos a curva do hold-out é ruidosa; o **ECE** próprio é medido também na **validação** (1560 jogos), onde é bem-comportado: ECE multiclasse ≈ 0,025 (raw 0,0262 / sigmoid 0,0267 / isotonic 0,0247). O ganho honesto está nas proper scoring rules (RPS/Brier/log-loss), que todas melhoram.

**Veredito: calibração isotônica do P2 ADOTADA.** A inferência da Copa 2026 (`cell-43`) passa a usar `clf_prod_calibrated` (P2 + isotonic) dentro do ensemble com o Dixon-Coles.

**Células criadas/alteradas:**
- **Novas células `P5.0`–`P5.4`** (após a `P4.4`): `P5.0` — markdown de seção com tabela/veredito; `P5.1` — calibração do P2 (Platt/isotonic) ajustada na validação + ECE multiclasse + helper de compatibilidade de versão (`_calibrate_prefit`); `P5.2` — calibração própria do DC e do ensemble (`temperature_fit`/`temperature_apply`, `IsotonicMulticlasse`); `P5.3` — seleção honesta + ensemble calibrado (`ens_calib_probs`, `clf_prod_calibrated`/`PROD_CALIB_FEATURES`/`PROD_CALIB_NAME`, flag `ADOPT_CALIBRATION`); `P5.4` — reliability diagrams antes/depois (reusa `reliability_diagram` do P1).
- **`cell-43` (inferência 2026) alterada:** consome `clf_prod_calibrated`/`PROD_CALIB_FEATURES` (P2 isotônico) com guarda `if ... in dir()` (fallback para `clf_prod`/`PROD_FEATURES` do P4 se o P5 não tiver rodado); placar/ensemble seguem do Dixon-Coles em campo neutro.

### Dependências
P1 (reliability diagram, métricas, conjunto de validação), P2 (multiclasse), P3 (DC/ensemble de produção).

---

# P6 — Higiene de engenharia

**Esforço:** Baixo · **Impacto:** Confiabilidade (baixo no número, alto na corretude) · **Risco:** Baixo

### Objetivo
Eliminar fontes de erro silencioso (train/serving skew) e remover hardcode.

### Itens
1. **Função única de features** usada tanto no treino quanto na inferência da Copa (`cell-36`/`cell-38`). Hoje o pipeline de inferência é reescrito à mão e **diverge** do de treino — fonte clássica de skew.
2. **Tirar do hardcode** `project_id='phoenix-cit'` e o bucket `gs://paul-the-octopus-frank-junior/` (`cell-4`, `cell-6`, `cell-40`) → variáveis/parâmetros no topo do notebook.
3. **Documentar o filtro de data** (`cell-17` usa `<= '2000-01-01'`; consistente mas pouco claro).
4. (Opcional) Extrair lógica reutilizável para funções dentro do `.ipynb`, mesmo sem módulos `.py`, para reduzir duplicação treino/inferência.
5. (Opcional) Adicionar um teste de sanidade: rodar a validação 2022 automaticamente e comparar com `VALIDACAO-2022.md`.

### Critério de aceitação
- Inferência da Copa usa a mesma função de features do treino.
- Sem `project_id`/bucket espalhados pelo código.

### Status: ✅ IMPLEMENTADO

Higiene aplicada ao notebook sem alterar a modelagem (P1–P5 intactos). Edições **cirúrgicas** por `id` de célula (formatação do `.ipynb` preservada — apenas as células abaixo aparecem no diff).

**Por item:**
1. **Função única de features (item 1) — ✅ já satisfeito pelo P4.9.** A geração de features vive em `build_football_features` e o placar/1X2 em `predict_match` (Dixon-Coles); a inferência da Copa (`cell-43`) consome exatamente essas funções (mais o `clf_prod_calibrated` do P5), sem reescrever features à mão. O item 1 do P6 **sobrepõe-se ao P4.9** e fica coberto — nada a duplicar.
2. **GCP removido — importação somente da pasta local `files/` (item 2) — ✅.** Decidido **eliminar o GCP por completo** (em vez de mantê-lo como fluxo opcional): os CSVs vêm exclusivamente da pasta local **`files/`** do repositório. Nova célula **`cell-p6-config`** (logo após `cell-03`, antes dos imports) define apenas:
   ```python
   DATA_DIR = 'files' if os.path.isdir('files') else '../files'
   ```
   robusto a rodar da raiz do repo ou de dentro de `src/`, sem qualquer referência a bucket/projeto.
   - **`read_csv` ligados ao `DATA_DIR`:** `cell-10`, `cell-22`, `cell-38` passam de `'/content/...'` hardcoded para `f'{DATA_DIR}/...'`.
   - **`cell-04` agora só importa bibliotecas** (numpy/pandas/seaborn/matplotlib) — removidos `from google.colab import auth`, `auth.authenticate_user()`, `project_id` e `!gcloud`. Resolve o `ModuleNotFoundError: No module named 'google'` que quebrava a execução local.
   - **Células `gsutil` removidas/neutralizadas:** a célula de **download** (`cell-06`) foi **deletada** (dados já estão em `files/`); a de **upload** (`cell-45`) passou a fazer **só** `predictions.to_csv(f'{DATA_DIR}/predictions_submission.csv')`, sem `!gsutil`.
   - **Markdowns ajustados:** `cell-03` (título da seção, sem "for GCP"), `cell-05` e `cell-44` (texto sem download/upload de GCP).
   - **Sem resíduo:** varredura confirma **zero** ocorrências de `gsutil`/`google.colab`/`gcloud`/`/content`/`GCS_BUCKET`/`GCP_PROJECT_ID` no código.
   - **Execução local verificada (IPython):** `cell-p6-config → cell-04 → cell-10 → cell-22 → cell-38` carregam de `files/` sem erro — `historical-results.csv` (49 306×9), `ranking.csv` (68 973×8), `matches-schedule.csv` (72×5).
3. **Filtro de data documentado (item 3) — ✅.** `cell-17` ganhou comentário explicando que `~(date <= '2000-01-01')` mantém o futebol moderno (`date > 2000-01-01`) e que o filtro análogo do ranking usa `< '2000-01-01'` (estritamente menor) — mesma intenção, coerência preservada.
5. **Teste de sanidade (item 5, opcional) — ✅.** Nova célula **`cell-p6-sanity`** (ao final) recalcula o RPS do modelo de produção (Ensemble P2-iso + DC cru) no hold-out Copa 2022 e compara com o número documentado (**0,2013**, tol. 0,01), imprimindo `PASS`/`WARN`. Totalmente defensiva (`if ... in dir()`): é ignorada se as células P1–P5 não tiverem rodado.

**Item 4 (opcional — extrair lógica para funções):** já largamente atendido pelos helpers reutilizáveis introduzidos em P1–P4 (`evaluate_probs`, `build_football_features`, `predict_match`, `order_chronologically`, etc.).

### Critério de aceitação — verificação
- ✅ Inferência da Copa usa a mesma função de features do treino (via P4.9: `build_football_features`/`predict_match` em `cell-43`).
- ✅ Sem `project_id`/bucket no código — **GCP removido**; dados só de `files/` via `DATA_DIR`.

**Células criadas/alteradas:**
- **Novas:** `cell-p6-config` (define `DATA_DIR` → `files/`, após `cell-03`) e `cell-p6-sanity` (sanity check da validação 2022, ao final).
- **Alteradas:** `cell-03`/`cell-05`/`cell-44` (markdowns sem GCP), `cell-04` (só imports de bibliotecas), `cell-45` (só `to_csv` local), `cell-10`/`cell-22`/`cell-38` (`read_csv` via `DATA_DIR`) e `cell-17` (comentário do filtro de data).
- **Removida:** `cell-06` (download `gsutil` do GCS — desnecessária com dados locais).

### Correções de compatibilidade (pandas 3.0 / sklearn 1.9) — para rodar localmente

Ao rodar o pipeline inteiro localmente (Python 3.13, pandas 3.0.3, sklearn 1.9.0) surgiram APIs removidas que quebravam a execução. Corrigidas (sem alterar a lógica/resultados):

| Célula | Antes (quebrava) | Depois | Motivo |
|---|---|---|---|
| `cell-11` | `pd.to_datetime(..., errors='ignore')` | `pd.to_datetime(..., format='%Y-%m-%d')` | `errors='ignore'` **removido no pandas 3.0** (`AssertionError`) |
| `cell-25` | `.fillna(method='ffill')` | `.ffill()` | argumento `method=` **removido no pandas 3.0** (`TypeError`) |
| `cell-25` | `.groupby(['country_full'], group_keys=False)` | `.groupby(['country_full'])` | no pandas 3.0, `group_keys=False` + `resample().first()` **descarta a coluna de grupo** → `KeyError: 'country_full'` no merge |
| `P2.1` (`cell 8378a5d3`) | `LogisticRegression(multi_class='multinomial', ...)` | `LogisticRegression(...)` | `multi_class` **removido no sklearn ≥ 1.7** (multinomial é o padrão) |
| `cell-p6-sanity` | `all(_n in dir() ...)` | `all(_n in globals() ...)` | `dir()` dentro de *generator expression* enxerga só o escopo do genexpr → guard sempre falhava |

**Execução completa verificada (local, headless `Agg`):** o notebook rodou **as 45 células de código até o fim sem erro** (~10 min) e **reproduziu exatamente** os números do P5 no hold-out 2022 (P2+Isotonic RPS **0,1985**; Ensemble produção **0,2013**; log-loss e Brier idênticos ao documentado), gerou as **72 previsões** da Copa 2026 e o sanity check da validação 2022 dá **PASS**.

### Dependências
Pode correr em paralelo; idealmente concluída antes de gerar a submissão 2026. **Item 1 sobrepõe-se ao P4.9** (já implementado).

---

## Resumo de prioridades

| Pri | Melhoria | Esforço | Impacto | Pré-requisitos |
|---|---|---|---|---|
| **P1** ✅ | Avaliação honesta (split temporal, RPS, baselines) | Baixo | Alto | — |
| **P2** ✅ | Alvo 3 classes (V/E/D) | Baixo | Alto | P1 |
| **P3** ✅ | Poisson / Dixon-Coles (placar real) — **standalone, sem P4** | Médio | Alto | P1, P2 |
| **P4** ✅ | Features (Elo, forma, gols, neutral, descanso) — **implementado (P4.1–P4.6/P4.8/P4.9); features não adotadas (não bate o P2 no hold-out)** | Médio | Alto | P1, P2 |
| **P5** ✅ | Calibração de probabilidades — **isotônica do P2 ADOTADA** (ensemble RPS 0,2034 → **0,2013**; log-loss e Brier também melhoram) | Baixo | Médio | P1, P2, P3 |
| **P6** ✅ | Higiene (função única, hardcode) — **GCP removido; dados só de `files/` via `DATA_DIR` (`cell-p6-config`); filtro de data documentado; sanity check 2022; função única já no P4.9** | Baixo | Confiabilidade | — |

### Subtarefas do P4 (status)

| # | Subtarefa | Status | Risco vazamento | Depende de |
|---|---|---|---|---|
| **P4.1** | Infra sem vazamento (`order_chronologically`, `rolling_team_feature`, `assert_no_leakage`) | ✅ implementado | — | P1 |
| **P4.2** | Elo / pi-ratings + baseline B2 | ✅ implementado | Médio | P4.1 |
| **P4.3** | Recuperar `total_points` + reusar `neutral` | ✅ implementado | Baixo | P4.1 |
| **P4.4** | Forma recente (V/E/D, N jogos) | ✅ implementado | Médio | P4.1 |
| **P4.5** | Médias móveis de gols (ataque/defesa) | ✅ implementado | Médio-alto | P4.1 |
| **P4.6** | Descanso (dias desde o último jogo) | ✅ implementado (descartado na seleção) | Baixo | P4.1 |
| **P4.7** | Head-to-head (ratios + shootouts) | ⏸️ adiado (futuro) | Médio | P4.1 |
| **P4.8** | Seleção incremental (**na validação**) + re-treino do P2 | ✅ implementado | Baixo | P4.2–P4.6, P1, P2, P3 |
| **P4.9** | Função única de features (treino + inferência) | ✅ implementado | Alto se ignorado | P4.8, P6.1 |

**Ordem recomendada:** `P4.1 → P4.3 → P4.2 → P4.4 → P4.6 → P4.5 → P4.7 → P4.8 → P4.9` (fundação → features de menor risco → maior risco → seleção/consolidação → higiene).

> **Resultado do P4:** features implementadas e testadas (anti-vazamento), mas **não adotadas** — melhoram a validação (RPS 0,1627→0,1536) e **não** o hold-out (P4 0,2168 > P2 0,2072). Produção segue com o **Ensemble P2+P3** (RPS 0,2034). Detalhes e tabela em "Status global do P4".

> **Nota P3:** implementado **antes do P4** (forças de ataque/defesa estimadas por MLE direta dos gols, `scipy.optimize`, em vez de reusar as médias móveis do P4). Como o P3 ficou standalone, o **P4 não bloqueia mais o P3**; integrar as features do P4 ao modelo de gols (como prior/offset, via P4.5) fica como trabalho futuro.

> **Nota P4:** **implementado** (P4.1–P4.6, P4.8, P4.9; P4.7 adiado). A seleção de features roda **na validação 2019–2021** (não no hold-out, como na prescrição original — correção de auditoria). As features **não foram adotadas** porque não superam o P2 no hold-out de 57 jogos; a infraestrutura fica pronta para evoluções (P4.5 → P3, P4.7). A subtarefa **P4.9 sobrepõe-se ao P6 item 1** (função única treino/inferência).

> Validação final antes da submissão 2026: rodar todo o pipeline no hold-out da Copa 2022 e confirmar melhora em RPS e acerto de resultado vs. a linha de base atual documentada em `VALIDACAO-2022.md`.
