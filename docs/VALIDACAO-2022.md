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

> Gerado a partir de `files/predictions_submission.csv` vs. resultados reais em `files/historical-results.csv` (FIFA World Cup, 2022).
