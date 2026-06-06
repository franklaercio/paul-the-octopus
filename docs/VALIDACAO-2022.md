# Validação do Modelo — Copa do Mundo 2022

Validação das predições geradas pelo modelo (`files/predictions_submission.csv`) contra os **resultados reais** da fase de grupos da Copa do Mundo FIFA 2022 (Catar), registrados em `files/historical-results.csv`.

## Critério de pontuação

| Situação | Pontos |
|---|---|
| Placar exato | **10** |
| Acertou o vencedor (ou empate), placar diferente | **5** |
| Errou o resultado | **0** |

> O empate conta como acerto de "vencedor" quando o modelo previu empate e o jogo terminou empatado.

## Resultado geral

| Métrica | Valor |
|---|---|
| **Pontuação total** | **140 / 480** (29.2%) |
| 🎯 Placar exato (10 pts) | 7 jogos = 70 pts |
| ✅ Vencedor certo (5 pts) | 14 jogos = 70 pts |
| ❌ Erro (0 pts) | 27 jogos |
| Total de partidas | 48 (fase de grupos) |
| Acerto do resultado (10+5) | 21/48 (43.8%) |

## Detalhe por partida

| # | Partida | Previsto | Real | Pontos |
|---|---|:-:|:-:|:-:|
| 1 | Qatar × Ecuador | 1–0 | 0–2 | ❌ 0 |
| 2 | Senegal × Netherlands | 0–1 | 0–2 | ✅ 5 |
| 3 | England × Iran | 0–1 | 6–2 | ❌ 0 |
| 4 | USA × Wales | 1–0 | 1–1 | ❌ 0 |
| 5 | France × Australia | 2–0 | 4–1 | ✅ 5 |
| 6 | Denmark × Tunisia | 2–0 | 0–0 | ❌ 0 |
| 7 | Mexico × Poland | 2–0 | 0–0 | ❌ 0 |
| 8 | Argentina × Saudi Arabia | 2–0 | 1–2 | ❌ 0 |
| 9 | Belgium × Canada | 2–0 | 1–0 | ✅ 5 |
| 10 | Spain × Costa Rica | 2–0 | 7–0 | ✅ 5 |
| 11 | Germany × Japan | 2–0 | 1–2 | ❌ 0 |
| 12 | Morocco × Croatia | 0–2 | 1–2 | ✅ 5 |
| 13 | Switzerland × Cameroon | 2–0 | 1–0 | ✅ 5 |
| 14 | Uruguay × South Korea | 0–2 | 0–0 | ❌ 0 |
| 15 | Portugal × Ghana | 0–1 | 3–2 | ❌ 0 |
| 16 | Brazil × Serbia | 2–0 | 2–0 | 🎯 10 |
| 17 | Wales × Iran | 0–1 | 0–2 | ✅ 5 |
| 18 | Qatar × Senegal | 0–2 | 1–3 | ✅ 5 |
| 19 | Netherlands × Ecuador | 0–2 | 1–1 | ❌ 0 |
| 20 | England × USA | 0–0 | 0–0 | 🎯 10 |
| 21 | Tunisia × Australia | 1–0 | 0–1 | ❌ 0 |
| 22 | Poland × Saudi Arabia | 2–0 | 2–0 | 🎯 10 |
| 23 | France × Denmark | 0–0 | 2–1 | ❌ 0 |
| 24 | Argentina × Mexico | 0–0 | 2–0 | ❌ 0 |
| 25 | Japan × Costa Rica | 0–0 | 0–1 | ❌ 0 |
| 26 | Belgium × Morocco | 2–0 | 0–2 | ❌ 0 |
| 27 | Croatia × Canada | 2–0 | 4–1 | ✅ 5 |
| 28 | Spain × Germany | 2–0 | 1–1 | ❌ 0 |
| 29 | Cameroon × Serbia | 0–2 | 3–3 | ❌ 0 |
| 30 | South Korea × Ghana | 0–0 | 2–3 | ❌ 0 |
| 31 | Brazil × Switzerland | 0–1 | 1–0 | ❌ 0 |
| 32 | Portugal × Uruguay | 2–0 | 2–0 | 🎯 10 |
| 33 | Wales × England | 0–1 | 0–3 | ✅ 5 |
| 34 | Iran × USA | 0–1 | 0–1 | 🎯 10 |
| 35 | Ecuador × Senegal | 0–0 | 1–2 | ❌ 0 |
| 36 | Netherlands × Qatar | 2–0 | 2–0 | 🎯 10 |
| 37 | Australia × Denmark | 0–2 | 1–0 | ❌ 0 |
| 38 | Tunisia × France | 0–2 | 1–0 | ❌ 0 |
| 39 | Poland × Argentina | 0–1 | 0–2 | ✅ 5 |
| 40 | Saudi Arabia × Mexico | 0–2 | 1–2 | ✅ 5 |
| 41 | Croatia × Belgium | 0–2 | 0–0 | ❌ 0 |
| 42 | Canada × Morocco | 2–0 | 1–2 | ❌ 0 |
| 43 | Japan × Spain | 2–0 | 2–1 | ✅ 5 |
| 44 | Costa Rica × Germany | 0–2 | 2–4 | ✅ 5 |
| 45 | Ghana × Uruguay | 0–2 | 0–2 | 🎯 10 |
| 46 | South Korea × Portugal | 0–2 | 2–1 | ❌ 0 |
| 47 | Serbia × Switzerland | 0–0 | 2–3 | ❌ 0 |
| 48 | Cameroon × Brazil | 0–2 | 1–0 | ❌ 0 |

## Observações

- O modelo acertou o **resultado** (vencedor/empate) em **21 de 48** jogos (44%).
- Apenas **7** placares exatos — esperado, já que a atribuição de placar é uma heurística por faixa de probabilidade (não um modelo de gols), tendendo a 2–0 / 0–2 / 0–0.
- Muitos erros (0 pts) vêm de **zebras** da Copa de 2022 (ex.: Arábia Saudita 2×1 Argentina, Japão sobre Alemanha e Espanha, Marrocos sobre Bélgica), que um modelo baseado só em ranking não captura.
- A acurácia de treino reportada (~94%) **não se traduz** nesta validação fora da amostra — sinal claro de superestimação da métrica de treino.

## Atualização P1 — avaliação honesta (números reais medidos)

A infraestrutura de avaliação honesta do P1 (split temporal + RPS/log-loss/Brier + baselines) foi implementada no notebook e mede o **classificador** (não a heurística de placar). Números reais (`random_state=42`, dados em `files/`), no hold-out das **57 partidas da Copa 2022** no dataset modelado (grupos + mata-mata):

| Métrica | Valor real |
|---|---|
| Acurácia binária in-sample (ilusão do notebook) | **93,27%** |
| Acurácia binária honesta (RandomForest, hold-out 2022) | **57,89%** |
| Acurácia binária (split temporal 80/20 genérico) | 67,07% |

| Modelo (1X2, hold-out 2022) | RPS ↓ | log-loss ↓ | Brier ↓ | acerto 1X2 |
|---|---|---|---|---|
| Modelo atual (RF binário, p_draw=0) | 0,3388 | 9,2124 | 0,9233 | 0,421 |
| B0 — frequência-base do treino | 0,2334 | 1,0777 | 0,6530 | 0,439 |
| B1 — favorito do ranking (suavizado) | 0,2019 | 1,0053 | 0,5982 | 0,579 |

O 94,24% confirmou-se ilusão de avaliação (treino = teste). O modelo binário perde em RPS até para o baseline trivial de frequência-base, porque nunca prevê empate (p_draw = 0) — motivação direta para o P2. Detalhes em [PLANO-MELHORIAS.md](PLANO-MELHORIAS.md#contexto-do-diagnóstico).

> A tabela "Detalhe por partida" acima refere-se à **heurística de placar** (fase de grupos, 48 jogos); as métricas do P1 avaliam o **classificador de probabilidades** sobre o hold-out completo da Copa 2022.

> Gerado a partir de `files/predictions_submission.csv` vs. resultados reais em `files/historical-results.csv` (FIFA World Cup, 2022).
