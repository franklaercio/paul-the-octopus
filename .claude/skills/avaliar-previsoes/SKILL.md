---
name: avaliar-previsoes
description: >-
  Avalia e pontua as previsões da Copa do Mundo 2026 do projeto Paul the Octopus,
  comparando data/results/predictions_submission.csv com os resultados reais em
  data/raw/worldcup-2026-results.csv. Use SEMPRE que o usuário quiser avaliar,
  pontuar, medir, validar ou "ver como está indo" o modelo/as previsões; comparar
  previsões com resultados reais; calcular acurácia, Brier, log-loss, RPS ou
  calibração; ou comparar contra baselines (sempre mandante, maior ranking FIFA) —
  mesmo que ele não diga a palavra "avaliação" explicitamente (ex.: "o Paul
  acertou os jogos?", "quão boas estão as previsões?", "o modelo bate o chute?").
---

# Avaliar previsões (Paul the Octopus)

Esta skill fecha o laço **prever → avaliar**: pega a submissão de previsões e a
confronta com o que de fato aconteceu na Copa 2026, usando as métricas certas
para previsão de futebol (1X2). Ela existe porque "acurácia" sozinha engana — um
modelo pode "acertar o resultado" e ainda assim estar mal calibrado ou perder
para um chute trivial. As métricas probabilísticas (Brier, log-loss, RPS) e a
calibração mostram a qualidade real, e os baselines dizem se vale mais que o
palpite óbvio.

## Como rodar (caminho principal)

A skill traz um pontuador pronto e testado — **use-o**, não reimplemente a
matemática das métricas a cada vez. A partir da raiz do repositório:

```bash
python .claude/skills/avaliar-previsoes/scripts/score_predictions.py
```

Ele descobre os caminhos padrão sozinho. Para sobrescrever, há flags:
`--predictions`, `--results`, `--schedule`, `--ranking`, `--historical`,
`--out-dir` e `--aliases` (JSON `{variante: canônico}` para nomes de seleção).

Saídas em `artifacts/` (não versionado):

- `avaliacao_jogo_a_jogo.csv` — uma linha por partida pontuada (probabilidades,
  placar real, previsto, acerto, e Brier/log-loss/RPS individuais).
- `avaliacao_resumo.json` e `avaliacao_resumo.md` — métricas acumuladas + baselines.
- `calibracao.png` — diagrama de confiabilidade.

Depois de rodar, **leia o resumo e explique ao usuário** o que os números dizem
(ver "Interpretação"). Não basta despejar a tabela.

## Contrato de entrada (importante)

O `03_predict.ipynb` ainda é um esqueleto, então **esta skill define o contrato**
de `data/results/predictions_submission.csv` — implemente o `03` para produzir
exatamente estas colunas:

| Coluna | Tipo | Significado |
|---|---|---|
| `match` | int | Número da partida; casa com `matches-schedule.csv` |
| `home`, `away` | str | Seleções (legibilidade e conferência) |
| `p_home`, `p_draw`, `p_away` | float | Probabilidades de 1/X/2; devem somar ~1 |

O pontuador renormaliza se a soma fugir de 1 (com aviso) e recusa probabilidades
ausentes/negativas. As entradas de referência são as do projeto:
`matches-schedule.csv` (`date` em DD/MM/AAAA, colunas `home`/`away`) e
`worldcup-2026-results.csv` (`date` em AAAA-MM-DD, colunas `home_team`/`away_team`).

## Reconciliação de schemas (o script já trata)

Os arquivos divergem de propósito, e o pontuador concilia:

- **Chave de junção**: previsões → calendário por `match`; calendário → resultados
  pelo **par ordenado de seleções** (na fase de grupos cada par joga uma vez, então
  o par é único e dispensa bater data em formatos diferentes).
- **Nomes de seleção**: normaliza (minúsculas, sem acentos) e aplica um mapa de
  apelidos (ex.: `USA`↔`United States`, `South Korea`↔`Korea Republic`) — relevante
  sobretudo para casar com `ranking.csv` (que usa a grafia da FIFA). Nomes que não
  casam são reportados, não engolidos.

## Métricas e baselines (resumo)

Detalhes, fórmulas e como interpretar cada uma: leia
[references/metricas.md](references/metricas.md).

- **Acurácia** — fração de resultados (1/X/2) em que o mais provável aconteceu. ↑ melhor.
- **Brier (multiclasse)** e **log-loss** — qualidade das probabilidades. ↓ melhor.
- **RPS** — score próprio para saídas **ordinais** (1 → X → 2): errar prevendo
  vitória do mandante num jogo que foi vitória do visitante pesa mais que prever
  empate. É a métrica mais indicada para 1X2. ↓ melhor.
- **Calibração** — quando o modelo diz 70%, acontece ~70% das vezes?

Baselines, para contexto: **taxa-base histórica** (frequências de 1X2 do histórico,
referência probabilística justa), **sempre mandante** e **maior ranking FIFA**
(palpites triviais, comparados por acurácia).

## Interpretação (estamos no meio do torneio)

- O teste de fogo não é "Brier baixo", e sim **superar a taxa-base** e bater os
  palpites triviais. Um modelo que não vence a taxa-base não aprendeu nada útil.
- Na fase de grupos só há resultado de um subconjunto das 72 partidas — o pontuador
  avalia só esse subconjunto e informa quantos jogos entraram. Com `n` pequeno, as
  métricas têm variância alta e a **calibração é ruidosa**; diga isso ao usuário em
  vez de cravar conclusões fortes.

## Se ainda não há previsões

Sem `predictions_submission.csv` não há o que pontuar. Nesse caso:

1. Diga claramente que falta a submissão (o `03_predict` é TODO).
2. Para destravar uma avaliação imediata, ofereça gerar um **baseline como
   submissão** (taxa-base ou maior ranking nas colunas do contrato acima) e pontuá-lo
   — assim já se mede o "piso" antes do modelo existir.

## Notas de conduta

- Avaliação compara com o passado já ocorrido; ainda assim, não deixe o conhecimento
  do resultado realimentar features do `01`/`02` (vazamento temporal).
- A skill é autocontida (pontuador próprio); ela **não** altera o pipeline. Se o
  usuário quiser a avaliação como artefato versionado do pipeline, aí sim proponha
  um `notebooks/04_evaluate.ipynb` que chame esta mesma lógica.
