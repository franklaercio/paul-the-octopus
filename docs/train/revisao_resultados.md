# Revisão crítica dos resultados — divergência validação × mundo real

> Análise da etapa 02 (treino) frente aos 20 jogos reais de 2026. Só diagnóstico;
> nenhuma alteração de código/notebook. Toda afirmação abaixo foi medida nos dados
> (`features.parquet`, `model.joblib`, `avaliacao_jogo_a_jogo.csv`).

## TL;DR

A divergência **é dominada por ruído de amostra (n=20)**, não por um defeito
estrutural do modelo. A perda para a taxa-base nos 20 jogos **não é
estatisticamente distinguível de zero** (bootstrap IC95% da diferença
modelo−base: **[−0,060, +0,087]**, cruza o zero; P(modelo pior)=65%). No
walk-forward (n=41.165) o modelo **vence a taxa-base em todas as fatias** e está
**bem calibrado** (gap confiança×acerto ≤2pp). O modelo não é superconfiante de
forma sistemática e não subestima empate de forma grosseira.

A causa imediata do número ruim nos 20 jogos é concreta e local: **40% dos 20
jogos terminaram empate (8/20)** contra 23,7% esperados, e **toda** a perda para
a taxa-base vem desses empates. Não há ação urgente. Há, sim, um ajuste de baixo
custo e baixo risco que protege contra exatamente esse cenário (regularização das
probabilidades em direção à taxa-base), que recomendo testar antes do `03`.

---

## 1. É ruído (n=20) ou sinal real? — Ruído, com forte evidência

**Quanto dá para concluir com 20 jogos: quase nada sobre o ranking modelo×base.**
Diferença observada de RPS (modelo−base) = **+0,0140**. Bootstrap pareado por
jogo (20.000 reamostragens):

| | valor |
|---|---|
| diferença média (modelo−base) | +0,0143 |
| IC95% | **[−0,0603, +0,0866]** |
| P(modelo pior que a base) | 65% |
| P(modelo melhor) | 35% |

O IC é ~6× mais largo que o ponto estimado e cruza o zero: com n=20 **não dá para
afirmar** que o modelo é pior que a taxa-base. O número real (0,2007) está dentro
do que se espera por flutuação amostral em torno do número de validação (0,175).

**Foram poucos erros pontuais ou viés sistemático? Poucos erros pontuais.**
Decomposição da diferença modelo−base por resultado real dos 20 jogos:

| resultado real | n | contribuição p/ (modelo−base) |
|---|---|---|
| **EMPATE** | 8 | **+0,716** (modelo muito pior) |
| home | 10 | +0,210 |
| away | 2 | **−0,646** (modelo muito melhor) |
| **total** | 20 | +0,280 |

Toda a perda está concentrada nos 8 empates; nos não-empates o modelo é melhor.
Os jogos que mais pesaram são palpites confiantes que bateram em empate:
Australia–Turkey, Côte d'Ivoire–Ecuador, **Spain–Cabo Verde (p_home 0,867 → 0×0)**,
**Iran–New Zealand (p_home 0,838 → 2×2)**, **Qatar–Switzerland (p_away 0,802 → 1×1)**.
São 5–6 jogos específicos, não um padrão difuso — assinatura de ruído, não de viés.

## 2. Diagnóstico provável

Testei as três hipóteses do enunciado contra o walk-forward (n=41.165, robusto):

| Hipótese | Veredito | Evidência |
|---|---|---|
| **Superconfiança / má calibração global** | **REJEITADA** | reliability no walk-forward: gap confiança−acerto ≤ 0,02 em todas as faixas (até 0,9). O modelo é honesto na escala grande. |
| **Subestima empate de forma grosseira** | **REJEITADA (na escala global)** | p_draw médio previsto 0,228 vs. empates reais 0,237 — casado. O "argmax nunca é empate" (1,8% no WF; 0/72 na submissão) é **correto** para 1X2: em jogos equilibrados as 3 probas ficam próximas e o empate raramente é o modo. Não é bug. |
| **Distribuição WC ≠ histórico** | **PARCIALMENTE VERDADEIRA, e é o fator real** | (a) **63/72 jogos de 2026 são neutros** vs. 26% do treino; (b) **tier 5 (Copa) é 100% dos 72** vs. 2% do treino; (c) os 72 confrontos são **mais desequilibrados** (\|elo_diff\| mediano 203 vs. 142). O modelo trata isso de forma plausível (coef de `is_neutral`: −0,37 home / +0,35 away), e nos jogos neutros do treino home/away quase se igualam (44%/33%), como esperado. |

**Diagnóstico mais provável (combinado):** o que aconteceu nos 20 jogos foi
**variância amostral de um torneio** (excesso de empates no início) batendo num
modelo que tem **uma fraqueza fina e conhecida**: nos **jogos equilibrados** ele
subestima empate um pouco mais que a média (walk-forward, \|elo_diff\|<75:
p_draw previsto 0,266 vs. real 0,284; a margem sobre a base cai para
0,2107 vs 0,2192). Copa tem muitos jogos equilibrados decididos no detalhe →
exatamente onde a fraqueza fina e o ruído de empate se encontram. **Não é
superconfiança global; é "déficit de empate em jogos parelhos" + azar de amostra.**

Fator agravante de produto: **o modelo não é calibrado** (`model_name: logreg`,
`class_weight: None`, sem `CalibratedClassifierCV`). A calibração global está boa
mesmo assim, mas calibração explícita é a alavanca mais barata para apertar o
déficit de empate nos jogos parelhos.

## 3. Recomendações priorizadas (o que eu faria, em ordem)

> Todas as correções abaixo foram **testadas nos 20 jogos** (in-sample, portanto
> otimistas — servem para confirmar direção, não para fixar hiperparâmetro). As
> três convergem para a mesma conclusão: encolher confiança / dar mais massa ao
> empate ajuda. Os valores ótimos devem ser fixados **na validação temporal**, não
> nestes 20.

**1º — Shrinkage das probabilidades em direção à taxa-base.** `P' = (1−α)·P + α·base`.
Mais barato, mais robusto, sem re-treinar. Nos 20 jogos: melhor α≈0,5–0,6 leva o
RPS de 0,2007 → **0,1764** (abaixo da base 0,1867). É um regularizador honesto que
protege contra o cenário "modelo confiante erra". **Fixar α por walk-forward**
(provável α modesto, 0,1–0,3, já que na escala grande o modelo é bom) e aplicá-lo
no `03`. Custo: ~5 linhas. Risco: mínimo.

**2º — Calibração explícita no `02` (já prevista no §7.F/§10.5, não implementada).**
Envolver o estimador em `CalibratedClassifierCV` com `cv` temporal e medir RPS
antes/depois (manter só se melhorar). Ataca a raiz do déficit de empate em vez de
mascará-lo. Equivalente a temperature scaling (T≈2 nos 20 jogos: 0,2007 → 0,1863).
Custo: baixo. Risco: baixo (decidir por delta de RPS, como o plano já manda).

**3º — Endereçar o empate em jogos equilibrados (a fraqueza fina real).** Opções,
em ordem de custo: (a) `class_weight` testado no logreg medindo RPS; (b) feature
de equilíbrio explícita (ex.: `abs(elo_diff)` ou um indicador de "jogo parelho")
para o modelo aprender que parelho ⇒ mais empate — hoje `is_neutral` quase não
move `draw` (coef +0,024); (c) a rota **Dixon-Coles** que o §7.A deixou registrada,
que corrige justamente o déficit de empate em placares baixos. Investir em DC só
se (a)/(b) não fecharem o gap — é a opção de maior custo. Testar na **fatia de
jogos equilibrados** do walk-forward, que é o termômetro certo.

**4º — Não mexer no tratamento de campo neutro.** Investiguei: `is_neutral` está
sendo aprendido e aplicado corretamente (coef coerente; jogos neutros do treino
têm home/away ~iguais; no walk-forward a fatia NEUTROS vence a base, 0,1846 vs
0,2371). O "viés de mando em 63 jogos neutros" **não** se confirmou como problema.
Aumento simétrico (§7.H) continua sendo experimento opcional, não prioridade.

**5º — Sobre concluir agora: aguardar mais jogos para o veredito, mas agir já no
barato.** Com n=20 não dá para dizer que o Paul "está ruim" — o sinal de validação
(n=41k, todas as fatias acima da base, bem calibrado) é muito mais confiável que o
ruído de 20 jogos. Reavaliar a divergência quando houver ~40–50 jogos reais
(fim da fase de grupos). Enquanto isso, **shrinkage + calibração** são ganhos de
custo-benefício alto que só fazem o modelo mais conservador onde ele precisa ser.

## O que eu faria primeiro

**Shrinkage para a taxa-base (α fixado por walk-forward) + calibração temporal no
`02`, decidida por delta de RPS.** Juntos, custam pouco, atacam o único modo de
falha observado (confiança excessiva pontual / déficit de empate em jogos
parelhos) e são reversíveis. Não tocar no campo neutro — está correto. Não tirar
conclusão definitiva sobre "modelo vs. base no mundo real" antes de ~40 jogos.

---

## Fechamento da iteração — veredito: CONGELAR em α=0,05

> Atualização após executar as recomendações (shrinkage + re-teste de calibração),
> com tuning honesto no walk-forward. Decisão tomada com o `football-data-scientist`.

A iteração confirmou o diagnóstico: **não há sinal exploitável barato a extrair.**

- **Shrinkage:** o α que minimiza o RPS walk-forward (≥1993, base por dobra,
  leak-free) é **α=0,05**, com ganho de apenas **0,00009** sobre α=0 — ruído
  numérico, não um ótimo; de α≥0,10 o RPS piora monotonicamente. Mantido
  `shrink_alpha=0,05` (é o argmin honesto) com o registro de que é **cosmético**
  (α=0 seria igualmente defensável por parcimônia). Não construir narrativa de
  "o shrinkage ajudou".
- **Calibração explícita:** isotônica/sigmoide **pioram** o RPS por delta →
  descartada (`calibrated=False`). Segunda confirmação de que o logreg já está calibrado.
- **20 jogos reais** (checagem out-of-sample, não usada para escolher): RPS
  0,2007 → 0,1969, Brier 0,6909 → 0,6821, log-loss 1,0908 → 1,0833 — melhoram um
  tico, mas seguem atrás da taxa-base (0,1867), **dentro do ruído** (IC95% cruza zero).

**Decisão: CONGELAR.** O lever de empate em jogos equilibrados (class_weight /
feature de equilíbrio / Dixon-Coles) **não é recomendado agora**: o gap é fino
(1,8pp numa fatia), `class_weight` tende a piorar o RPS global, e Dixon-Coles é
reescrita de paradigma (**backlog v2**, não para esta Copa).

**Gate de reabertura (objetivo):** ao atingir ~40–50 jogos reais (fim dos grupos),
refazer o bootstrap pareado modelo−base; só reabrir modelagem se o IC95% passar a
**excluir o zero do lado ruim**. Enquanto cruzar zero, o walk-forward (n=41k)
prevalece e o modelo fica congelado.

**Risco de fundo:** a divergência real é de distribuição (88% neutros / 100%
tier-Copa no predict vs 26% / 2% no treino) — limitação de dados, não de modelo;
nenhum lever de empate a resolve. A resposta certa é mais jogos, não mais ajuste.

---

### Apêndice — números-chave

- Submissão (72): p_home 0,456 / p_draw 0,219 / p_away 0,325 (médias); p_draw teto
  0,329; argmax = home 45×, away 27×, **draw 0×**; pmax mediano 0,624; 11 jogos
  com pmax>0,8.
- 20 jogos reais: 10 home / **8 draw** / 2 away (empate observado 40% vs base 22,7%).
- Walk-forward (n=41.165): RPS modelo 0,1754 vs base 0,2254; reliability gap ≤0,02;
  RPS por desfecho real: home 0,123 / draw 0,184 / **away 0,259** (away é o mais
  difícil, não o empate).
- Modelo: `LogisticRegression` (C=1,0, `class_weight=None`), **não calibrado**;
  coef `is_neutral` = −0,37 (home) / +0,35 (away) / +0,02 (draw).
- Distribuição: neutros 26% (treino) → 88% (predict); tier 5 = 2% (treino) → 100%
  (predict); \|elo_diff\| mediano 142 (treino) → 203 (predict).
