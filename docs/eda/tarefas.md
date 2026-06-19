# Tarefas — EDA (`00_eda.ipynb`)

Checklist acionável do plano em [`README.md`](README.md). Agrupado pelas etapas
E0–E6 (= seções §5.1–§5.7). **Não usar `worldcup-2026-results.csv` em nenhuma.**

## E0 — Ingestão e panorama (§5.1)
- [ ] Estender a carga para incluir `shootouts.csv` e
      `historical_win-loose-draw_ratios.csv` (além dos 3 já carregados).
- [ ] Conferir `dtypes` e parse de datas (`date`, `rank_date`); corrigir tipos.
- [ ] Gerar `info()` / `describe()` / cardinalidade dos categóricos.
- [ ] Montar o **dicionário de dados** (coluna, tipo, exemplo, % nulos) por arquivo.

## E1 — Auditoria de qualidade (§5.2)
- [ ] Nulos por coluna em cada arquivo.
- [ ] Duplicatas: linha inteira e por chave `date+home+away` no histórico.
- [ ] Faixas plausíveis de `home_score`/`away_score` (≥0, sem absurdos); datas válidas.
- [ ] Consistência do flag `neutral`.
- [ ] `shootouts`: `winner ∈ {home_team, away_team}`.
- [ ] `ranking`: coerência `rank` × `total_points` e do `rank_change`.
- [ ] `h2h`: `wins+looses+draws ≈ 1` e simetria dos pares dirigidos.
- [ ] Escrever o **relatório de qualidade** + decisões de tratamento.

## E2 — Reconciliação de chaves (§5.3)
- [ ] Normalizar nomes (minúsculas, sem acento) e aplicar `DEFAULT_ALIASES` da
      skill `avaliar-previsoes` (não criar outro mapa).
- [ ] Resolver explicitamente `Iran`→"IR Iran" e `South Korea`→"Korea Republic".
- [ ] Listar nomes não resolvidos em histórico/ranking/shootouts/h2h.
- [ ] Mapear confederação das 48 seleções de 2026.
- [ ] Produzir a **tabela canônica de seleções** (canônico, apelidos, confederação,
      ranking recente).

## E3 — Distribuições univariadas (§5.4)
- [ ] Distribuição de gols: mandante, visitante e total.
- [ ] Taxas de resultado 1/X/2 (geral e só competitivo).
- [ ] Volume por década e mix de torneios (amistoso vs. competitivo).
- [ ] Proporção de jogos em campo neutro.
- [ ] Ranking: distribuição de pontos, `rank_change`, pontos por confederação.
- [ ] Calendário 2026: grupos, horários, sedes.
- [ ] Auxiliares: pênaltis no tempo; cobertura de pares no `h2h`.
- [ ] Exportar figuras-chave para `docs/eda/figuras/`.

## E4 — Sinais preditivos (§5.5)
- [ ] Vantagem de mando: 1/X/2 mandante vs. visitante vs. **campo neutro**; tendência.
- [ ] Gols vs. Poisson: dispersão, taxa de empate observada vs. prevista,
      correlação mandante×visitante.
- [ ] Ranking: P(vitória) e saldo de gols vs. diferença de ranking (≥1993).
- [ ] Forma recente (janela móvel) — separa o desfecho? (nota de vazamento).
- [ ] Viabilidade de **Elo pré-jogo**: separação dos desfechos pela diferença de Elo.
- [ ] H2H: agrega além de ranking/Elo? medir esparsidade.
- [ ] Amistoso vs. competitivo: diferença de desfecho/gols → ponderar/filtrar?
- [ ] Fadiga/calendário: dias de descanso entre jogos como feature candidata.
- [ ] Escrever a **lista ranqueada de sinais** com evidência + nota de vazamento.

## E5 — Aplicabilidade ao 2026 (§5.6)
- [ ] Cobertura por seleção (jogos nos últimos N anos).
- [ ] H2H disponível por confronto (dos 72).
- [ ] Equilíbrio por confederação.
- [ ] Definir tratamento de mando/neutro para 2026 (anfitriões vs. demais).
- [ ] Montar a **tabela de prontidão** com sinalização de escassez.

## E6 — Síntese e ponte p/ `01` (§5.7)
- [ ] Consolidar achados num **memo** curto (`docs/eda/achados.md` e/ou no notebook).
- [ ] Fixar as taxas-base a superar (conectar aos baselines de `avaliar-previsoes`).
- [ ] Montar a **tabela de features candidatas** (definição + regra "só pré-jogo").
- [ ] Listar riscos e decisões em aberto.
- [ ] Rodar o checklist de Definition of Done (§9 do README).

## Conferência final
- [ ] Notebook reexecuta de cima a baixo sem erro (`Restart & Run All`).
- [ ] Nenhuma célula lê `worldcup-2026-results.csv`.
- [ ] `00_eda.ipynb` continua **fora** do `run_pipeline`.
