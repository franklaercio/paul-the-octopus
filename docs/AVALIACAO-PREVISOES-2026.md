# Avaliação das Previsões — Copa do Mundo 2026 (fase de grupos)

Validação de **plausibilidade** das 72 previsões em `files/predictions_submission.csv`, geradas pelo modelo de produção **Ensemble (P2-isotonic + Dixon-Coles)** (RPS 0,2013 no hold-out 2022). Acompanha o plano de ajuste fino do notebook `src/paultheoctopus.ipynb` na seção final.

> **Natureza desta validação.** A Copa 2026 ainda não começou (início em 11/06/2026; este documento é de 06/06/2026). Não há resultado real para comparar — diferente de [VALIDACAO-2022.md](VALIDACAO-2022.md), que mede contra o ocorrido. Aqui avaliamos **validade aparente** (face validity): se cada placar previsto é coerente com a força relativa das seleções, com o que se espera de um torneio em campo neutro e com as distribuições históricas de gols/resultados. Os números de qualidade (RPS/Brier/log-loss) continuam vindo do hold-out 2022 documentado nos outros relatórios.

---

## 1. Diagnóstico agregado (as 72 partidas)

| Métrica | Previsões 2026 | Referência histórica | Leitura |
|---|---:|---:|---|
| Resultado: mandante (calendário) | 41,7% | 48,7% (com mando) | ok — campo neutro reduz o "mandante" |
| Resultado: **empate** | **29,2%** | 21,7% (dataset) / ~22–25% (Copas) | ⚠️ **inflado** |
| Resultado: visitante (calendário) | 29,2% | 29,5% | ok |
| **Gols por jogo (total)** | **1,32** | **~2,5–2,7** (Copas recentes; 2022 = 2,69) | 🔴 **metade do real** |
| Gols por time/jogo | 0,66 | ~1,25–1,35 | 🔴 comprimido |
| Jogos com ≥3 gols totais | 4 / 72 (5,6%) | ~35–45% nas Copas | 🔴 quase ausente |
| Maior nº de gols de um time | **3** | 5–7 não é raro numa Copa | 🟡 cauda alta sumiu (artefato da moda) |
| Placares previstos | 1-0 (17), 0-1 (16), 1-1 (11), 2-0 (10), 0-0 (10), 0-2 (4), 3-0 (3), 0-3 (1) | — | tudo no "cluster baixo" |

**Conclusão do agregado (corrigida após medição no hold-out):** o **vencedor (1X2) é plausível**. A baixa contagem de gols **não é** um defeito da média do modelo — é um **artefato de reportar a moda**. Medido no hold-out 2022:

| Quantidade | Valor | Leitura |
|---|---:|---|
| **E[gols/jogo] do modelo Dixon-Coles** (média, λ_casa+λ_fora) | **2,10** | razoável — só ~19% abaixo dos 2,60 reais |
| Gols/jogo do **placar reportado** (argmax = moda) | **0,98** | comprimido **pela natureza da moda**, não pela média |
| Gols/jogo reais (hold-out 2022) | 2,60 | referência |

O notebook escolhe o placar pelo **argmax** da matriz (o placar isolado mais provável). Para Poisson com média ~1 gol por time, a **moda** fica em 0/1 gol mesmo com a média em ~1,05 por time — então o placar reportado colapsa para 1-0 / 0-0 / 1-1. Isso explica os dois desvios visíveis (poucos gols **e** empates a mais), **mas a média de gols do modelo está em ordem certa**. Reportar um único placar exige escolher um ponto da distribuição, e a moda é o que maximiza o acerto exato.

> **Ponto crítico, comprovado empiricamente (ver §4 P7.1):** sob o critério de pontuação da submissão (10 placar exato / 5 resultado / 0 erro), o **argmax é quase-ótimo** — rende **195 pts no hold-out**, melhor que a alternativa "placar de pontos esperados" (180 pts). Ou seja: os placares baixos da submissão **não são um bug a corrigir para pontuar** — são a aposta que maximiza pontos dado o modelo atual. O verdadeiro espaço de melhora não está na *seleção* do placar (P7.1, rejeitado), e sim na *calibração da média de gols* do DC (P7.2) e no desempate de grupos (P7.5).

---

## 2. Validação por grupo

Legenda: ✅ coerente · ⚠️ defensável, mas com ressalva · 🔴 implausível/artefato.

### Grupo A — Czech Republic, South Korea, Mexico, South Africa
- ✅ Ordenação por ranking plausível (Czech Republic à frente).
- 🔴 **5 dos 6 jogos com 0 ou 1 gol e três empates** (Mexico 0-0 South Africa, South Korea 1-1 Czech, Mexico 1-1 South Korea, South Africa 1-1 South Korea). Grupo "todo empatado" — classificação decidida no desempate alfabético (Mexico × South Africa ambos 2 pts, SG −1). **Artefato da compressão de placar**, não sinal real.

### Grupo B — Switzerland, Bosnia, Canada, Qatar
- ✅ Switzerland 100% (9 pts) é coerente.
- 🔴 Bosnia, Canada e Qatar **todos com 2 pts**, separados por critério alfabético/confronto. Canada 0-0 Qatar e Bosnia 0-0 Qatar reforçam a inflação de empate entre médios.

### Grupo C — Turkey, Paraguay, USA, Australia
- ⚠️ **Três seleções empatadas em 5 pts** (Turkey, Paraguay, USA: 1V 2E). Plausível que sejam parelhas, mas a incapacidade de separá-las vem de novo dos muitos empates 0-0/1-1.
- ✅ Australia com 0 pts no fundo é coerente.

### Grupo D — Brazil, Morocco, Scotland, Haiti
- ⚠️ **Brazil 0-0 Morocco** e Brazil campeão do grupo com só **4 GP em 3 jogos**. Morocco é forte (semifinalista 2022), então um empate é defensável; o problema é Brazil marcar tão pouco no agregado (3-0 vs. Haiti é o único jogo "normal").
- ✅ Haiti 0 pts, 0 GP, −5 — coerente com minnow.

### Grupo E — Germany, Ecuador, Côte d'Ivoire, Curaçao
- ⚠️ **Ecuador 1-1 Germany** e Ecuador empatado em pontos com a Germany (7). Ecuador tem ranking decente; defensável, mas tende a superestimar o time médio por causa do empate fácil.
- ✅ Curaçao 0 pts no fundo.

### Grupo F — Netherlands, Sweden, Tunisia, Japan
- 🔴 **Japan no fundo (4º, 1 pt)** abaixo de Tunisia. Pelo futebol recente o Japão (que bateu Alemanha e Espanha em 2022) está acima da Tunísia — aqui o ranking/Elo histórico não captura a ascensão japonesa. Sinal de **feature de forma recente faltando** (P4.4 foi implementado mas não adotado).
- ✅ Netherlands e Sweden avançando é coerente.

### Grupo G — Belgium, Iran, Egypt, New Zealand
- ✅ Belgium 1º coerente.
- ⚠️ Iran 2º com **dois empates 1-1** (vs. Belgium e Egypt). Iran segurando a Bélgica em campo neutro é possível, mas é mais um empate "de compressão".

### Grupo H — Spain, Uruguay, Cabo Verde, Saudi Arabia
- ✅ Spain 9 pts e Uruguay 6 — totalmente coerente.
- ✅ Cabo Verde e Saudi Arabia no fundo (1 pt cada). Bom contraste — aqui o modelo **separa bem** porque há favoritos claros.

### Grupo I — France, Norway, Senegal, Iraq
- ✅ France 100%, Iraq 0 — coerente.
- ⚠️ Norway 2º à frente de Senegal por critério (ambos 4 pts, SG 0). Senegal (campeão africano) ≥ Norway é discutível; Norway sobe pelo Elo de elenco. Defensável.

### Grupo J — Argentina, Algeria, Austria, Jordan
- ✅ Argentina 100%, **6 GP (o melhor ataque previsto)** — o favorito mais "premiado" do quadro, coerente.
- ⚠️ Algeria × Austria empatados (4 pts); desempate fraco.

### Grupo K — Portugal, Colombia, Congo DR, Uzbekistan
- ✅ Portugal 9 pts/6 GP e Colombia 6 — coerentes.
- ✅ Uzbekistan 0 pts/−6 no fundo — coerente.

### Grupo L — England, Croatia, Ghana, Panama
- ✅ England 9 pts, Panama 0/−6 — coerentes.
- ✅ Croatia 2ª — coerente com ranking.

**Resumo da validação por grupo:** os **extremos** (favoritos fortes e minnows) saem **corretos e bem separados** (Grupos H, I, K, L). A degradação aparece no **meio da tabela** (Grupos A, B, C), onde a compressão de placar gera blocos de 3 seleções empatadas resolvidos por critério alfabético — sem significado esportivo. Há **um possível erro de mérito** (Japan abaixo de Tunisia, Grupo F) atribuível à ausência de forma recente.

---

## 3. Problemas sistêmicos (causas-raiz)

1. **🟡 Placar reportado pela moda (argmax).** O placar é a moda da matriz, que para médias ~1 gol/time fica em 0/1 gol — daí 0,98 gol/jogo no placar reportado **embora** a média do modelo seja 2,10 (§1). **Não é, sozinho, um defeito a corrigir:** medido no hold-out sob 10/5/0, o argmax é quase-ótimo (195 pts) e bate a alternativa "pontos esperados" (P7.1, **rejeitado** — ver §4). É uma limitação *intrínseca* de reportar um placar único, não um bug de seleção.
2. **🔴 Média de gols do DC levemente baixa.** A causa de fundo dos placares baixos é a **média**: E[gols/jogo] = 2,10 vs. 2,60 reais (−19%). Subir a média do DC (P7.2) empurra a própria moda para cima (1-0 → 1-1/2-1) e é o lever mais promissor — mas precisa passar na regra de ouro do hold-out.
3. **⚠️ Empates inflados (29,2% nas previsões 2026).** Sintoma combinado: média de gols um pouco baixa + campo neutro (γ=0 no DC, sem desempate de mando) → para jogos parelhos o placar modal vira 0-0/1-1. Resultado: blocos de seleções tecnicamente empatadas, classificação decidida por critério alfabético.
4. **⚠️ Sem forma recente / só força histórica.** O modelo de produção usa ranking + Elo histórico (as features P4 de forma/gols **não foram adotadas** por não baterem o P2 no hold-out de 57 jogos). Casos como **Japan < Tunisia** sugerem que a forma do ciclo 2022–2026 não está representada.
5. **⚠️ Desempate de grupo é frágil.** [PREVISAO-GRUPOS-2026.md](PREVISAO-GRUPOS-2026.md) resolve empates por SG → GP → confronto direto → **ordem alfabética**. Com 21 empates previstos, vários 2º/3º lugares (e os "melhores terceiros") dependem do critério alfabético — pouco robusto para uma submissão.
6. **ℹ️ Seeding do mata-mata não é o oficial da FIFA.** O chaveamento em [PREVISAO-GRUPOS-2026.md](PREVISAO-GRUPOS-2026.md) é por tier/desempenho, não o slotting real. Afeta o campeão previsto (Brazil), não a fase de grupos — registrado para transparência.

---

## 4. Plano de ajuste fino do notebook (`src/paultheoctopus.ipynb`)

Etapa **P7 — Geração de placar orientada à pontuação**, alinhada à metodologia dos P1–P6 (toda mudança aceita só se **melhora o RPS/placar no hold-out 2022** contra a linha de base atual). Ordenadas por impacto/esforço.

### P7.1 — Placar por pontos esperados *(IMPLEMENTADO e ❌ REJEITADO)*
- **Onde:** helper `expected_points_scoreline` na célula `P3.2`; avaliação na `P3.3`.
- **O quê:** em vez de `argmax P(i,j)` (placar modal), escolher o placar `(i,j)` que **maximiza o ganho esperado sob o critério da submissão** (10 placar exato / 5 resultado / 0 erro): `E[pts] = 10·P(exato) + 5·(P(resultado) − P(exato))`, usando a matriz inteira. Testadas duas variantes: marginal 1X2 do próprio DC e marginal reescalada para o ensemble (P2-iso + DC).
- **Resultado real (hold-out 2022, critério 10/5/0):**

  | Método de placar | Pontos (10/5/0) | Placar exato |
  |---|---:|---:|
  | **argmax / modal (produção)** | **195** | **8/57** |
  | P7.1 — marginal ensemble | 180 | 5/57 |
  | P7.1 — marginal DC | 165 | 5/57 |

- **Veredito: ❌ REJEITADO.** O argmax **vence** sob o próprio critério da submissão (195 > 180 > 165) e acerta mais placares exatos (8 vs. 5). A maximização de pontos esperados é teoricamente ótima *se o modelo fosse bem calibrado em gols*; como o DC tem média um pouco baixa e cauda imperfeita, a troca sacrifica acertos exatos (nível 10 pts) sem repor o suficiente no nível 5 pts. Pela regra de ouro do projeto, **não é adotado** — a produção mantém o argmax. O helper e a medição ficam no notebook (`P3.2`/`P3.3`) como experimento documentado, no mesmo padrão do P4 rejeitado.

### P7.2 — Calibrar a média de gols do Dixon-Coles *(lever principal — médio esforço)*
- **Onde:** `fit_dixon_coles` (`P3.1`) / `predict_match` (`P3.2`).
- **O quê:** a causa de fundo dos placares baixos **não** é a seleção do placar (P7.1), e sim a **média de gols** do DC: medido, **E[gols/jogo] = 2,10 vs. 2,60 reais** (−19%). Subir a média (corrigir o intercepto de gols / revisar `Σα = 0` / reavaliar o decaimento temporal e o `ρ` low-score) empurra **a própria moda** para cima (1-0 → 1-1/2-1), atacando placar baixo **e** empate inflado de uma vez — diferente do P7.1, que só remexe a seleção. Reportar **λ esperado** junto do placar para diagnóstico contínuo (já impresso na `P3.3`).
- **Validar:** E[gols/jogo] no hold-out → ~2,5–2,6 (±10%); **placar exato ≥ 8/57 e pontos 10/5/0 ≥ 195** (não pode regredir vs. argmax atual); RPS do 1X2 não piora.

### P7.3 — Auditar a inflação de empate *(médio impacto)*
- **Onde:** `P3.2`/`P3.3` e a inferência.
- **O quê:** comparar a **taxa de empate dos placares** (29,2% na Copa 2026) com a massa de empate que o 1X2 atribui (DC ≈ 27,7%) e com os empates reais (24,6% no hold-out). O descolamento é, em parte, o mesmo efeito da moda baixa — deve **diminuir** com o P7.2 (mais gols ⇒ menos 0-0/1-1 modais). Medir antes/depois do P7.2.
- **Validar:** taxa de empate dos placares ≈ massa de empate do 1X2.

### P7.4 — Margem flexível para empate *(implementado e adotado)*
- **Onde:** `choose_outcome_with_draw_margin` / `_scorelines_draw_margin` (`P7.4`).
- **O quê:** o condicionamento rígido ao 1X2 quase nunca deixava o placar terminar empatado, mesmo quando `p_draw` estava competitivo. A nova regra permite empate quando `p_draw >= max(p_home, p_away) - DRAW_MARGIN_PROD`.
- **Validado:** `DRAW_MARGIN_PROD=0.08` elevou o hold-out de **200 → 220 pts**, com **8 → 9 placares exatos**, **32 → 35 resultados** e empates de **1,8% → 10,5%** no hold-out.

### P7.5 — Desempate de grupo principiado *(médio esforço, robustez)*
- **Onde:** geração de `PREVISAO-GRUPOS-2026.md` (script de tabela) — não altera o modelo.
- **O quê:** substituir o último critério (**alfabético**) por **probabilidade de avanço via simulação Monte Carlo**: amostrar N placares da matriz Dixon-Coles de cada jogo, montar a tabela em cada simulação e reportar **P(1º), P(2º), P(avançar)** por seleção. Resolve os blocos empatados de forma esportiva e dá incerteza honesta.
- **Validar:** nenhum grupo decidido por ordem alfabética; ranking dos "melhores terceiros" por probabilidade, não por desempate frágil.

### P7.6 — Reabrir forma recente como feature *(testado/rejeitado para produção)*
- **Onde:** `build_football_features` (`P4.2`) + seleção (`P4.3`/`P4.4`).
- **O quê:** o P4 foi reaberto com features Last10 de ataque/defesa. A validação melhorou, mas o hold-out piorou, então a feature não entrou no classificador final. No placar, o ajuste de lambdas com Last10 também foi rejeitado: o melhor ponto de validação foi `k=0`.
- **Validar:** regra de ouro inalterada (só adota se RPS melhora no hold-out).

### P7.7 — Higiene da saída *(baixo esforço)*
- O arquivo `predictions_submission.csv` duplicado na **raiz do repositório** (além de `files/`) deve ser removido/consolidado — a `cell-45` grava em `files/` (ver P6). Garantir fonte única.

### Resumo de prioridades P7

| Pri | Ajuste | Status | Esforço | Impacto | Onde |
|---|---|---|---|---|---|
| **P7.1** | Placar = max pontos esperados (não argmax) | ❌ **testado e rejeitado** (195→180 pts no hold-out) | Baixo | — | `P3.2`/`P3.3` |
| **P7.2** | Calibrar a média de gols do DC (2,10→~2,6) | ✅ **adotado** (`goal_scale=1.35`) | Médio | **Alto** | `P3.1`/`P3.2` |
| **P7.3** | Last10 modulando lambdas do placar | ❌ **testado e rejeitado** (`k=0`) | Médio | — | `P3.2`/`P7.3` |
| **P7.4** | Margem flexível para empate | ✅ **adotado** (`DRAW_MARGIN_PROD=0.08`; 200→220 pts) | Baixo | Alto | `P3.3`/`P7.4` |
| **P7.5** | Desempate de grupo por Monte Carlo | proposto | Médio | Médio (robustez) | script de grupos |
| **P7.6** | Reabrir forma recente no classificador | ❌ **testado e rejeitado** no hold-out | Médio | — | `P4.2`–`P4.4` |
| **P7.7** | Remover CSV duplicado na raiz | proposto | Baixo | Higiene | repo / `cell-45` |

---

## 5. Experimento Last10 — ataque/defesa recente

Implementado no notebook (`P4.2`-`P4.4`) conforme o plano em
[`PLANO-FEATURES-ATAQUE-DEFESA-LAST10.md`](PLANO-FEATURES-ATAQUE-DEFESA-LAST10.md).

**O que foi adicionado:**

- `gf_last10_home` / `gf_last10_away`
- `ga_last10_home` / `ga_last10_away`
- `attack_diff_last10`
- `defense_diff_last10`
- `home_pressure_last10`
- `away_pressure_last10`
- `pressure_diff_last10`
- versoes ajustadas por adversario/competicao:
  - `attack_diff_last10_adj`
  - `defense_diff_last10_adj`
  - `pressure_diff_last10_adj`

Todas as features usam apenas jogos anteriores a partida atual (`shift(1)` +
janela movel de 10 jogos), considerando mandante e visitante em uma visao longa
por selecao. As versoes ajustadas ponderam gols marcados/sofridos pela forca do
adversario via ranking FIFA (`rank_ref / rank_opp`, com clipping) e dao peso um
pouco maior a Copa do Mundo. O rolling tambem ignora linhas futuras da Copa 2026
como atualizacao historica, evitando que placares placeholder contaminem outras
partidas futuras.

O teste anti-vazamento do P4 foi ampliado para incluir as features last10 e
passou: alterar artificialmente o placar da propria partida para `9-0` nao muda
as features dessa partida.

**Resultado medido:**

| Etapa | Resultado |
|---|---:|
| Melhor last10 isolada na validacao | `attack_diff_last10_adj` |
| RPS validacao P2 base | 0,1627 |
| RPS validacao P4 com last10 ajustado selecionado | 0,1535 |
| Last10 selecionada pelo greedy | `attack_diff_last10_adj` |
| RPS hold-out P2 | 0,2072 |
| RPS hold-out P4 + last10 ajustado | 0,2172 |

**Veredito:** o ajuste por adversario/competicao aumentou o sinal na validacao,
mas piorou a generalizacao no hold-out Copa 2022. Pela regra de ouro do projeto,
o P4 com last10 ajustado foi **rejeitado** e o modelo de producao segue sendo
**P2 + Isotonic (calibrado) + Dixon-Coles**.

**Teste adicional no modelo de gols (P7.3):** tambem testamos usar
`home_pressure_last10_adj` / `away_pressure_last10_adj` como modulador suave dos
lambdas do Dixon-Coles:

```text
lambda_home *= clip(exp(k * home_pressure_last10_adj), lo, hi)
lambda_away *= clip(exp(k * away_pressure_last10_adj), lo, hi)
```

Grid testado na validacao:

```text
k in {0.00, 0.03, 0.05, 0.08, 0.10}
clip in {0.90-1.10, 0.85-1.15, 0.80-1.20}
```

Resultado: o melhor ponto foi `k=0.00`, isto e, **sem ajuste last10 nos
lambdas**. Qualquer `k > 0` reduziu a pontuacao 10/5/0 na validacao. No
hold-out, o candidato `k=0` empata com a base por ser a propria base:

| Metodo | Validacao pontos | Validacao exatos | Hold-out pontos | Hold-out exatos |
|---|---:|---:|---:|---:|
| Base sem lambda last10 | 6490 | 265 | 200 | 8 |
| Melhor grid (`k=0`) | 6490 | 265 | 200 | 8 |
| Melhor `k > 0` | 6460 | 259 | - | - |

**Veredito P7.3:** rejeitado. A feature last10 ajustada parece carregar sinal
local, mas quando aplicada aos gols esperados ela desloca massa de placares de
um jeito que piora acerto exato na validacao.

**Condicionamento flexivel ao empate (P7.4):** como o condicionamento rigido ao
`argmax` do ensemble deixava poucos empates, testamos permitir empate quando a
probabilidade de empate esta perto do melhor resultado:

```text
se p_draw >= max(p_home, p_away) - margin -> placar de empate
senao -> placar condicionado ao argmax 1X2
```

Grid testado na validacao:

```text
margin in {0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15}
```

Resultado:

| Metodo | Validacao pontos | Validacao exatos | Hold-out pontos | Hold-out exatos | Hold-out resultado |
|---|---:|---:|---:|---:|---:|
| Base (`margin=0.00`) | 6490 | 265 | 200 | 8 | 32 |
| Melhor grid (`margin=0.08`) | 6535 | 268 | 220 | 9 | 35 |

**Veredito P7.4:** adotado. A margem `DRAW_MARGIN_PROD=0.08` melhora validacao
e hold-out, recupera empates sem reabrir as zebras artefatuais e passa na regra
de ouro do projeto.

---

## 6. Veredito

- **1X2 (vencedor/empate): coerente.** A ordenação dos grupos respeita a força das seleções; favoritos (Spain, France, Argentina, Portugal, England, Brazil) e minnows (Curaçao, Panama, Uzbekistan, Iraq, Jordan, Haiti) saem nos lugares esperados. Nada *implausível* no pódio dos grupos.
- **Placar (gols): melhorou após P7.2 + P7.4.** A produção atual usa `goal_scale=1.35` no Dixon-Coles e condiciona o placar ao 1X2 do ensemble com margem flexivel de empate `DRAW_MARGIN_PROD=0.08`. O arquivo atual fica em **2,17 gols/jogo**, **12 placares únicos**, **13/72 empates** e **3 jogos 0-0**. O P7.1 por pontos esperados segue rejeitado; a melhoria adotada foi calibrar a média de gols, travar coerência com o 1X2 e permitir empate quando competitivo.
- **Empates sairam do extremo baixo.** Antes da P7.4 eram 5/72; agora sao 13/72. Ainda abaixo da faixa historica de Copas (~22-25%), mas muito menos artificial e validado por hold-out.
- **Forma recente last10 foi testada e rejeitada para produção.** `attack_diff_last10_adj` entrou no greedy por validação, mas o P4 + last10 ajustado piorou no hold-out; logo fica documentado como experimento, não como modelo de produção.

> Os números de qualidade probabilística (RPS 0,2013 no hold-out 2022) permanecem válidos. A produção segue como **Ensemble P2-iso + DC cru para 1X2**, com placar vindo do **Dixon-Coles escalado (`goal_scale=1.35`)**, `lambda_last10_k=0.00` e `DRAW_MARGIN_PROD=0.08`. O notebook roda de ponta a ponta e os CSVs de `files/` estão alinhados ao calendário.

---

> Gerado a partir de `files/predictions_submission.csv` e `files/schedule_with_predictions.csv` (72 partidas da fase de grupos 2026). Referências de qualidade: [PLANO-MELHORIAS.md](PLANO-MELHORIAS.md), [VALIDACAO-2022.md](VALIDACAO-2022.md), [PREVISAO-GRUPOS-2026.md](PREVISAO-GRUPOS-2026.md).
