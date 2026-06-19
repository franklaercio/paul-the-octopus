# Plano de Execução — EDA (`notebooks/00_eda.ipynb`)

Plano da exploração de dados (EDA) do Paul the Octopus. A EDA é **exploratória e
fica fora do pipeline executável** (`run_pipeline.py` ignora o `00_eda.ipynb`):
ela não grava artefatos consumidos pelas etapas seguintes. Seu produto é
**entendimento + recomendações** que orientam a engenharia de features no
`01_features.ipynb`.

> **Escopo de dados — exclusão deliberada:** a EDA usa todos os CSVs de
> `data/raw/` **exceto `worldcup-2026-results.csv`**. Esses resultados são o
> *gabarito* de avaliação (ver a skill `avaliar-previsoes`); mantê-los fora da
> exploração evita que decisões de modelagem sejam contaminadas pelo que já
> aconteceu na Copa (vazamento). O `historical-results.csv` vai até 2026-06-10
> (véspera do Mundial), então amistosos pré-Copa entram normalmente — só os jogos
> do torneio (de 11/06 em diante) ficam de fora.

---

## 1. Objetivos

1. **Conhecer a estrutura e a qualidade** de cada dataset (tipos, nulos,
   duplicatas, faixas, consistência) antes de qualquer modelagem.
2. **Resolver as chaves de junção** entre os arquivos — sobretudo a grafia das
   seleções, que diverge entre histórico, ranking e calendário.
3. **Estabelecer as taxas-base** que qualquer modelo terá de superar (vantagem de
   mando, taxa de empate, distribuição de gols) — as mesmas referências usadas
   pelos baselines da skill `avaliar-previsoes`.
4. **Mapear sinais preditivos candidatos** (ranking FIFA, força tipo Elo, forma
   recente, confronto direto, mando/campo neutro, fadiga/calendário) e medir, de
   forma exploratória, quanto cada um separa os desfechos.
5. **Avaliar a aplicabilidade ao 2026**: quanta história existe para as 48
   seleções e os 72 confrontos, e onde os dados são escassos.
6. **Entregar uma ponte explícita para o `01_features`**: lista de features
   candidatas com definição e a regra de "só informação pré-jogo", além dos
   riscos a vigiar.

## 2. Dados em escopo

| Arquivo | Linhas | Período | Colunas-chave | Uso na EDA |
|---|---|---|---|---|
| `historical-results.csv` | 49.402 | 1872 → 2026-06-10 | `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` | Núcleo: alvo, gols, mando, forma, Elo |
| `ranking.csv` | 68.973 | 1992 → 2026-04 | `rank, country_full, country_abrv, total_points, ..., confederation, rank_date` | Sinal de força; só ≥1993 |
| `matches-schedule.csv` | 72 | 2026 | `match, date, time_brasilia, timezone, home, away, phase` | Alvo da previsão (48 seleções) |
| `shootouts.csv` | 677 | 1967 → | `date, home_team, away_team, winner` | Auxiliar: desempate/contexto |
| `historical_win-loose-draw_ratios.csv` | 798 | — | `home, away, games, wins, looses, draws` (pares dirigidos) | Auxiliar: confronto direto |
| ~~`worldcup-2026-results.csv`~~ | — | 2026-06-11 → | — | **Excluído** (gabarito) |

## 3. Princípios e guardrails

- **Anti-vazamento temporal.** A EDA não toca no `worldcup-2026-results.csv`. Ao
  esboçar qualquer sinal (forma, Elo, ranking), use apenas informação disponível
  **antes** da partida — isso vale aqui e, com força total, no `01_features`.
- **Reprodutibilidade.** Semente fixa, código determinístico, caminhos resolvidos
  via a função `find_root()` que já existe no notebook.
- **Sem efeitos colaterais no pipeline.** A EDA não grava em `data/processed/`,
  `models/` nem `data/results/`. Figuras-chave podem ser exportadas para
  `docs/eda/figuras/` (registro versionado); o grosso dos gráficos fica inline.
- **Uma fonte de verdade para nomes.** Reaproveite o mapa de apelidos
  `DEFAULT_ALIASES` de
  `.claude/skills/avaliar-previsoes/scripts/score_predictions.py` em vez de criar
  outro — assim EDA, features e avaliação concordam sobre quem é quem.

## 4. Plano geral (etapas)

A EDA roda em 7 etapas; cada etapa é uma seção do notebook (§5) e tem suas tarefas
em [`tarefas.md`](tarefas.md). A ordem importa: qualidade e reconciliação **antes**
de qualquer análise relacional, porque junções erradas produzem conclusões erradas.

| Etapa | Seção | Entrega |
|---|---|---|
| **E0** Ingestão e panorama | §5.1 | Dicionário de dados por arquivo |
| **E1** Auditoria de qualidade | §5.2 | Relatório de qualidade + decisões |
| **E2** Reconciliação de chaves | §5.3 | Tabela canônica de seleções |
| **E3** Distribuições univariadas | §5.4 | Figuras + taxas-base |
| **E4** Sinais preditivos (alvo) | §5.5 | Sinais candidatos ranqueados |
| **E5** Aplicabilidade ao 2026 | §5.6 | Prontidão por confronto/seleção |
| **E6** Síntese e ponte p/ `01` | §5.7 | Lista de features + memo de achados |

## 5. Tópicos

Os eixos transversais que a EDA precisa cobrir:

- **Qualidade** — nulos, duplicatas, tipos, faixas, sentinelas, datas.
- **Estrutura e chaves** — como os CSVs se conectam; reconciliação de nomes.
- **Distribuições** — gols, resultados, ranking, ao longo do tempo e por contexto.
- **Relações orientadas ao alvo** — o que separa vitória/empate/derrota e gols.
- **Contexto de futebol** — mando vs. campo neutro, amistoso vs. competitivo,
  fadiga/viagem, confederação, era.
- **Aplicabilidade ao 2026** — cobertura e suficiência de dados.
- **Riscos** — vazamento, esparsidade, ruído de amistosos, início do ranking (1993).

## 6. Seções e análises (detalhado)

### §5.1 — Ingestão e panorama (E0)
**Objetivo:** carregar os 5 arquivos com tipos/datas corretos e ter um panorama.
**Análises:** `shape`, `dtypes`, `info()`, `head()`, uso de memória; `describe()`
para numéricos e cardinalidade para categóricos (`tournament`, `confederation`).
**Perguntas:** os tipos vieram certos? datas parseadas? volume por arquivo?
**Saída:** um **dicionário de dados** (uma tabela por arquivo: coluna, tipo,
exemplo, % nulos).

### §5.2 — Auditoria de qualidade (E1)
**Objetivo:** achar problemas antes que virem conclusões falsas.
**Análises:** nulos por coluna; duplicatas (linha inteira e por chave de partida
`date+home+away`); faixas plausíveis (`home_score`/`away_score` ≥ 0 e não
absurdos); datas futuras/impossíveis; consistência do `neutral`; em `shootouts`,
`winner ∈ {home_team, away_team}`; em `ranking`, coerência entre `rank` e
`total_points` e do `rank_change`; em `h2h`, `wins+looses+draws ≈ 1` e simetria
dos pares dirigidos.
**Perguntas:** o que limpar, o que manter, o que sinalizar?
**Saída:** **relatório de qualidade** (tabela de checagens) + lista de decisões
de tratamento.

### §5.3 — Reconciliação de chaves / entidades (E2)
**Objetivo:** garantir que "Brazil" é a mesma seleção em todos os arquivos.
**Contexto medido:** as 48 seleções de 2026 aparecem **48/48** no histórico, mas
**46/48** no ranking — faltam `Iran` ("IR Iran") e `South Korea` ("Korea
Republic"). **Análises:** normalizar nomes (minúsculas, sem acento) e aplicar o
`DEFAULT_ALIASES` da skill; listar nomes não resolvidos; mapear confederação das
48; checar cobertura também em `shootouts` e `h2h`.
**Saída:** **tabela canônica de seleções** (nome canônico, apelidos,
confederação, ranking mais recente) e a lista de pendências de nomenclatura.

### §5.4 — Distribuições univariadas (E3)
**Objetivo:** conhecer cada variável isoladamente.
**Análises:**
- *Histórico:* distribuição de gols (mandante, visitante, total); taxas de
  resultado 1/X/2; volume por década (cresce muito pós-1990); mix de torneios
  (amistosos dominam: 18.389) ; proporção de jogos em campo neutro (~26%).
- *Ranking:* distribuição de `total_points` e `rank_change`; trajetórias de
  exemplo; pontos por confederação.
- *Calendário:* composição dos grupos, horários (todos GMT-3), sedes.
- *Auxiliares:* frequência de pênaltis no tempo; cobertura de pares no `h2h`.
**Saída:** figuras + **taxas-base** numéricas (1/X/2 geral e só-competitivo).

### §5.5 — Análise orientada ao alvo: sinais candidatos (E4)
**Objetivo:** medir, de forma exploratória, o que prediz o desfecho (1X2 e gols).
**Análises:**
- **Vantagem de mando:** taxas 1/X/2 mandante vs. visitante e **em campo neutro**
  (crítico: jogos de grupo de 2026 são em sedes fixas, muitos neutros para quem
  não é anfitrião); tendência por década.
- **Gols e Poisson:** ajustar gols a uma Poisson; medir sub/superdispersão e a
  taxa de empate observada vs. a prevista; correlação entre gols de mandante e
  visitante (motiva ajuste tipo Dixon-Coles em placares baixos).
- **Ranking como sinal:** P(vitória) vs. diferença de ranking; diferença de
  ranking vs. saldo de gols (restrito a ≥1993).
- **Forma recente:** janelas móveis de resultados/gols antes do jogo — separam o
  desfecho? (esboço, atento a vazamento).
- **Força tipo Elo (viabilidade):** esboçar um Elo pré-jogo a partir do histórico
  e ver a separação dos desfechos pela diferença de Elo. *Exploração*, não o
  modelo final.
- **Confronto direto (H2H):** o `h2h` agrega sinal além de ranking/Elo? qual a
  esparsidade?
- **Amistoso vs. competitivo:** diferenças de desfecho/gols → implicação de
  ponderar ou filtrar amistosos.
- **Fadiga/calendário:** dias de descanso entre jogos a partir das datas →
  viabilidade como feature.
**Saída:** **lista ranqueada de sinais candidatos** com a evidência de separação e
uma nota de vazamento para cada.

### §5.6 — Aplicabilidade ao 2026 (E5)
**Objetivo:** saber se há dados suficientes onde o modelo vai prever.
**Análises:** cobertura por seleção (jogos nos últimos N anos); H2H disponível por
confronto (quantos dos 72 têm jogos prévios entre as duas); equilíbrio por
confederação; tratamento de mando/neutro para 2026 (anfitriões EUA/Canadá/México
com mando; demais em campo neutro).
**Saída:** **tabela de prontidão** por confronto/seleção, com sinalização de
escassez (ex.: estreantes ou seleções com pouca história recente).

### §5.7 — Síntese e ponte para o `01_features` (E6)
**Objetivo:** transformar a exploração em decisões.
**Análises:** consolidar achados; fixar as taxas-base a superar (conectar com os
baselines da skill `avaliar-previsoes`); montar a lista de features candidatas com
definição operacional e a regra "só pré-jogo"; listar riscos (vazamento,
esparsidade, ruído de amistosos, ranking só ≥1993, nomes, campo neutro) e decisões
em aberto.
**Saída:** **tabela de features candidatas** (entrada do `01`) + um **memo de
achados** curto.

## 7. Entregáveis

- `notebooks/00_eda.ipynb` preenchido nas 7 seções acima.
- Figuras-chave em `docs/eda/figuras/` (as principais; o resto fica inline).
- Memo de achados + tabela de features candidatas (no fim do notebook e/ou em
  `docs/eda/achados.md`).
- Tabela canônica de seleções (reaproveitável por `01`/avaliação).

## 8. Mapeamento para o notebook

O `00_eda.ipynb` hoje tem: célula de *setup* (`find_root`, caminhos), uma célula
que carrega `historico/ranking/calendario`, um markdown "O que explorar" e um
`# TODO`. O plano **estende** isso:

- Acrescentar à carga `shootouts` e `historical_win-loose-draw_ratios`.
- Substituir o `# TODO` pelas seções §5.1–§5.7, cada uma com um cabeçalho markdown
  seguido das células de análise.
- Manter o notebook fora do `run_pipeline` (nada de gravar artefatos do pipeline).

## 9. Critérios de conclusão (Definition of Done)

- [ ] Todo arquivo em escopo tem dicionário de dados e relatório de qualidade.
- [ ] 48/48 seleções de 2026 resolvidas para nome canônico em todos os arquivos.
- [ ] Taxas-base 1/X/2 (geral e competitivo) e distribuição de gols documentadas.
- [ ] Cada sinal candidato tem evidência de separação + nota de vazamento.
- [ ] Tabela de prontidão do 2026 e lista de features candidatas escritas.
- [ ] Nenhuma célula lê `worldcup-2026-results.csv`.

> Tarefas acionáveis e granulares: ver [`tarefas.md`](tarefas.md).
