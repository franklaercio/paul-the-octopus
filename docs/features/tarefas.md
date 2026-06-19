# Tarefas — Features (`01_features.ipynb`)

Checklist acionável do plano em [`README.md`](README.md), por etapas E0–E7
(= seções §6.0–§6.7). **Nunca usar `worldcup-2026-results.csv`.**

## E0 — Prep, ambiente e reconciliação (§6.0)
- [ ] Garantir ambiente com `pyarrow` (`pip install -r requirements-dev.txt`).
- [ ] Carregar histórico, ranking, calendário (+ auxiliares se necessário).
- [ ] Aplicar `canon()` (mapa `DEFAULT_ALIASES` da skill) a todas as seleções.
- [ ] Deduplicar o confronto duplicado (`date+home+away`) achado na EDA.
- [ ] Tratar os 21 `rank` nulos do ranking.
- [ ] Montar a **tabela longa por seleção-jogo** em ordem cronológica.

## E1 — Núcleo de força: Elo + ranking (§6.1)
- [ ] Calcular **Elo pré-jogo** (registrar antes, atualizar depois; HFA só se não-neutro).
- [ ] Juntar **ranking *as-of*** (último rank estritamente antes da data).
- [ ] Derivar `elo_diff`, `rank_diff`, `points_diff`.

## E2 — Mando/neutro e contexto 2026 (§6.2)
- [ ] Definir `is_neutral` no histórico.
- [ ] Para 2026: mando só aos anfitriões (EUA/Canadá/México); demais neutro.

## E3 — Forma e força ofensiva/defensiva (§6.3)
- [ ] Forma recente: pontos e gols pró/contra em janela móvel **deslocada** (`shift`).
- [ ] Força ofensiva/defensiva por seleção (média móvel de gols).
- [ ] (Opcional) ponderar por recência.

## E4 — Descanso/congestionamento (§6.4)
- [ ] `rest_days_*` no histórico (da tabela longa).
- [ ] `rest_days_*` para 2026 (do calendário).

## E5 — Confronto direto e confederação (§6.5)
- [ ] H2H com `h2h_winrate_home`, `h2h_games`, `h2h_available` (flag de cobertura).
- [ ] Confederação como categórica (`confed_home`, `confed_away`).
- [ ] Codificar `tournament`: reduzir a baldes de domínio + ordinal de importância (não `0,1,2` cru); one-hot fica no `02`.

## E6 — Montagem do `features.parquet` (§6.6)
- [ ] Computar as features para **treino** (histórico) e **previsão** (72 jogos) com a mesma rotina.
- [ ] Anexar `target_outcome`, `home_score`/`away_score` e `sample_weight` (só treino).
- [ ] Adicionar `split` e `match_no`; gravar no contrato da §2.

## E7 — Validação e testes (§6.7)
- [ ] Acrescentar validação de `features.parquet` ao `run_pipeline.py`
      (schema; sem NaN nas chaves; 72 linhas `predict` cobrindo o calendário;
      alvo presente sse `split=="train"`).
- [ ] Testes em `tests/` (schema + **smoke test de vazamento**).
- [ ] Ajustar o contrato do `03` para ler `features.parquet` (`split=="predict"`).
- [ ] `ruff check` + `pytest` + `run_pipeline` verdes.

## Conferência final
- [ ] Notebook reexecuta de cima a baixo sem erro (`Restart & Run All`).
- [ ] Nenhuma célula lê `worldcup-2026-results.csv`.
- [ ] `features.parquet` bate com o contrato e alimenta `02`/`03`.
