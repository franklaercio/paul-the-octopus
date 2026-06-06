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

> **Ponto crítico, comprovado empiricamente (ver §4 P7.1):** sob o critério de pontuação da submissão (10 placar exato / 5 resultado / 0 erro), o **argmax é quase-ótimo** — rende **195 pts no hold-out**, melhor que a alternativa "placar de pontos esperados" (180 pts). Ou seja: os placares baixos da submissão **não são um bug a corrigir para pontuar** — são a aposta que maximiza pontos dado o modelo atual. O verdadeiro espaço de melhora não está na *seleção* do placar (P7.1, rejeitado), e sim na *calibração da média de gols* do DC (P7.2) e no desempate de grupos (P7.4).

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

### P7.4 — Desempate de grupo principiado *(médio esforço, robustez)*
- **Onde:** geração de `PREVISAO-GRUPOS-2026.md` (script de tabela) — não altera o modelo.
- **O quê:** substituir o último critério (**alfabético**) por **probabilidade de avanço via simulação Monte Carlo**: amostrar N placares da matriz Dixon-Coles de cada jogo, montar a tabela em cada simulação e reportar **P(1º), P(2º), P(avançar)** por seleção. Resolve os blocos empatados de forma esportiva e dá incerteza honesta.
- **Validar:** nenhum grupo decidido por ordem alfabética; ranking dos "melhores terceiros" por probabilidade, não por desempate frágil.

### P7.5 — Reabrir forma recente como feature *(médio esforço, trabalho futuro)*
- **Onde:** `build_football_features` (`P4.2`) + seleção (`P4.3`/`P4.4`).
- **O quê:** o P4 foi rejeitado no hold-out de **57 jogos** (amostra pequena). Antes da submissão final, re-testar **só `form_diff` + `elo_diff`** (as de maior sinal) com validação em janela maior (ex.: 2019–2021 **e** 2022 como validação estendida, reservando outro hold-out), mirando casos como **Japan < Tunisia**. Adotar **apenas** se melhorar o RPS fora da amostra — manter o rigor que rejeitou o P4.
- **Validar:** regra de ouro inalterada (só adota se RPS melhora no hold-out).

### P7.6 — Higiene da saída *(baixo esforço)*
- O arquivo `predictions_submission.csv` duplicado na **raiz do repositório** (além de `files/`) deve ser removido/consolidado — a `cell-45` grava em `files/` (ver P6). Garantir fonte única.

### Resumo de prioridades P7

| Pri | Ajuste | Status | Esforço | Impacto | Onde |
|---|---|---|---|---|---|
| **P7.1** | Placar = max pontos esperados (não argmax) | ❌ **testado e rejeitado** (195→180 pts no hold-out) | Baixo | — | `P3.2`/`P3.3` |
| **P7.2** | Calibrar a média de gols do DC (2,10→~2,6) | proposto (**lever principal**) | Médio | **Alto** | `P3.1`/`P3.2` |
| **P7.3** | Auditar inflação de empate | proposto | Baixo | Médio | `P3.3` |
| **P7.4** | Desempate de grupo por Monte Carlo | proposto | Médio | Médio (robustez) | script de grupos |
| **P7.5** | Reabrir forma recente (Japan < Tunisia) | proposto | Médio | Médio | `P4.2`–`P4.4` |
| **P7.6** | Remover CSV duplicado na raiz | proposto | Baixo | Higiene | repo / `cell-45` |

---

## 5. Veredito

- **1X2 (vencedor/empate): coerente.** A ordenação dos grupos respeita a força das seleções; favoritos (Spain, France, Argentina, Portugal, England, Brazil) e minnows (Curaçao, Panama, Uzbekistan, Iraq, Jordan, Haiti) saem nos lugares esperados. Nada *implausível* no pódio dos grupos.
- **Placar (gols): baixo, mas pela média do DC — não pela seleção do placar.** O placar reportado dá 1,32 gol/jogo porque é a **moda**, enquanto a média do modelo é 2,10 (vs. 2,60 reais). O **P7.1** (placar por pontos esperados) foi **implementado, medido e rejeitado**: sob o critério real 10/5/0 o argmax atual rende **195 pts** vs. 180 do P7.1 no hold-out. O lever certo é **P7.2** (calibrar a média de gols do DC), que ataca placar baixo e empate inflado juntos.
- **Empates inflados (29,2%)** e **blocos de grupo no alfabético**: parte é a média de gols baixa (P7.2), parte é o desempate frágil (P7.4).
- **Único ponto de atenção de mérito:** Japan abaixo de Tunisia — falta de forma recente (P7.5).

> Os números de qualidade probabilística (RPS 0,2013 no hold-out 2022) permanecem válidos e **a produção segue inalterada** (argmax + Ensemble P2-iso + DC): o P7.1 foi rejeitado pela própria regra de ouro do projeto, sem regredir a submissão. O notebook ainda roda de ponta a ponta (sanity check 2022 = **PASS**).

---

> Gerado a partir de `files/predictions_submission.csv` e `files/schedule_with_predictions.csv` (72 partidas da fase de grupos 2026). Referências de qualidade: [PLANO-MELHORIAS.md](PLANO-MELHORIAS.md), [VALIDACAO-2022.md](VALIDACAO-2022.md), [PREVISAO-GRUPOS-2026.md](PREVISAO-GRUPOS-2026.md).
