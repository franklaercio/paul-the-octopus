# Métricas de avaliação de previsões 1X2

Referência das métricas usadas pelo pontuador. Notação: para cada partida há um
vetor de probabilidades previstas `p = (p_home, p_draw, p_away)` e o resultado
real como vetor one-hot `o` (ex.: empate → `(0, 1, 0)`). As saídas são tratadas
como **ordinais** na ordem `home → draw → away`.

## Acurácia

Fração de partidas em que `argmax(p)` coincide com o resultado real. ↑ melhor.

Intuitiva, mas grosseira: ignora a confiança e trata "60% mandante" e "99%
mandante" como iguais. Serve para comunicação, não como métrica principal.

## Brier (multiclasse)

```
BS = média_n  Σ_k (p_nk − o_nk)²
```

Erro quadrático entre probabilidade e desfecho, somado nas 3 classes e
promediado nas partidas. Intervalo `[0, 2]`; **↓ melhor**. É uma *proper scoring
rule*: minimiza-se dizendo a verdade, então não recompensa exageros de confiança.

## Log-loss (entropia cruzada)

```
LL = média_n  −log(p_correto)   (com clipping de p para evitar log(0))
```

Penaliza fortemente errar com confiança (prever 1% no que aconteceu ≈ punição
enorme). ↓ melhor. Também é proper. É mais sensível a casos extremos que o Brier;
por isso o clipping (limita o estrago de um zero).

## RPS — Ranked Probability Score

```
RPS = média_n  ( 1/(r−1) · Σ_{i=1}^{r−1} ( CDF_p(i) − CDF_o(i) )² )
```

com `r = 3` classes e `CDF` a soma acumulada ao longo de `home → draw → away`.
Intervalo `[0, 1]`; **↓ melhor**.

É a métrica mais indicada para 1X2 porque respeita a **ordem** dos desfechos:
prever vitória do mandante num jogo que terminou em vitória do visitante é um erro
"de duas casas" e pesa mais do que ter previsto empate (erro "de uma casa"). Brier
e log-loss não enxergam essa distância ordinal; o RPS sim.

## Calibração (confiabilidade)

Um modelo é calibrado quando, entre os eventos a que atribuiu ~`x`% de chance, a
frequência observada também é ~`x`%. O pontuador agrupa as probabilidades em
faixas (abordagem *one-vs-rest*: cada par partida×classe vira uma previsão binária
`p` vs `aconteceu?`) e plota probabilidade média prevista × frequência observada.
A diagonal é a calibração perfeita; pontos acima = subconfiança, abaixo =
superconfiança.

Cuidado: com poucas partidas (início/meio de torneio) as faixas ficam quase vazias
e o diagrama vira ruído. Trate-o como indício, não veredito.

## Baselines

Métrica sozinha não diz se o modelo é bom — bom **comparado a quê?**

- **Taxa-base histórica** — frequências de 1/X/2 no histórico (`historical-results.csv`)
  aplicadas igualmente a todos os jogos. Referência probabilística justa: dá para
  comparar Brier/log-loss/RPS. Não vencer a taxa-base = o modelo não agregou sinal.
- **Sempre mandante** — prevê vitória do mandante sempre; comparado por acurácia.
- **Maior ranking FIFA** — prevê a seleção melhor colocada no ranking (último
  `rank_date` por seleção em `ranking.csv`); comparado por acurácia.

Os dois últimos são "chutes triviais": se o modelo não os supera em acurácia, ele
não está pagando o próprio custo.
