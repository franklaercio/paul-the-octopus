# Previsão dos Grupos — Copa do Mundo FIFA 2026 (Paul the Octopus)

Compilação da **fase de grupos** a partir das previsões do modelo de produção (**Ensemble P2-isotonic + Dixon-Coles**, RPS 0,2013 no hold-out 2022). Placar de cada jogo gerado pelo modelo Dixon-Coles (argmax da matriz de placares).

**Fontes:** `files/matches-schedule.csv` (calendário) + `files/predictions_submission.csv` (placar previsto) → `files/schedule_with_predictions.csv`.

> **Como os grupos foram reconstruídos:** o calendário não rotula os grupos; foram inferidos pelos confrontos (times do mesmo grupo se enfrentam em turno único — 4 times, 6 jogos). Os rótulos A–L seguem a ordem de aparição no calendário.

> **Critérios de desempate:** pontos → saldo de gols (SG) → gols pró (GP) → confronto direto → ordem alfabética (último recurso). Com muitos empates previstos, há blocos tecnicamente empatados — ver nota em cada grupo.

Legenda: **J** jogos · **V** vitórias · **E** empates · **D** derrotas · **GP** gols pró · **GC** gols contra · **SG** saldo · **P** pontos. ✅ = classificado direto (1º/2º). 🟡 = 3º (pode avançar como melhor terceiro).

## Grupo A

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Czech Republic ✅ | 7 | 3 | 2 | 1 | 0 | 3 | 1 | +2 |
| 2 | South Korea ✅ | 3 | 3 | 0 | 3 | 0 | 3 | 3 | +0 |
| 3 | Mexico 🟡 | 2 | 3 | 0 | 2 | 1 | 1 | 2 | -1 |
| 4 | South Africa | 2 | 3 | 0 | 2 | 1 | 1 | 2 | -1 |

**Jogos:**

- 11/06/2026 — Mexico 0–0 South Africa
- 11/06/2026 — South Korea 1–1 Czech Republic
- 18/06/2026 — Czech Republic 1–0 South Africa
- 18/06/2026 — Mexico 1–1 South Korea
- 24/06/2026 — Mexico 0–1 Czech Republic
- 24/06/2026 — South Africa 1–1 South Korea

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo B

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Switzerland ✅ | 9 | 3 | 3 | 0 | 0 | 3 | 0 | +3 |
| 2 | Bosnia and Herzegovina ✅ | 2 | 3 | 0 | 2 | 1 | 1 | 2 | -1 |
| 3 | Canada 🟡 | 2 | 3 | 0 | 2 | 1 | 1 | 2 | -1 |
| 4 | Qatar | 2 | 3 | 0 | 2 | 1 | 0 | 1 | -1 |

**Jogos:**

- 12/06/2026 — Canada 1–1 Bosnia and Herzegovina
- 13/06/2026 — Qatar 0–1 Switzerland
- 18/06/2026 — Switzerland 1–0 Bosnia and Herzegovina
- 18/06/2026 — Canada 0–0 Qatar
- 24/06/2026 — Canada 0–1 Switzerland
- 24/06/2026 — Bosnia and Herzegovina 0–0 Qatar

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo C

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Turkey ✅ | 5 | 3 | 1 | 2 | 0 | 3 | 2 | +1 |
| 2 | Paraguay ✅ | 5 | 3 | 1 | 2 | 0 | 2 | 1 | +1 |
| 3 | USA 🟡 | 5 | 3 | 1 | 2 | 0 | 2 | 1 | +1 |
| 4 | Australia | 0 | 3 | 0 | 0 | 3 | 0 | 3 | -3 |

**Jogos:**

- 12/06/2026 — USA 0–0 Paraguay
- 13/06/2026 — Australia 0–1 Turkey
- 19/06/2026 — USA 1–0 Australia
- 19/06/2026 — Turkey 1–1 Paraguay
- 25/06/2026 — USA 1–1 Turkey
- 25/06/2026 — Paraguay 1–0 Australia

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo D

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Brazil ✅ | 7 | 3 | 2 | 1 | 0 | 4 | 0 | +4 |
| 2 | Morocco ✅ | 5 | 3 | 1 | 2 | 0 | 1 | 0 | +1 |
| 3 | Scotland 🟡 | 4 | 3 | 1 | 1 | 1 | 1 | 1 | +0 |
| 4 | Haiti | 0 | 3 | 0 | 0 | 3 | 0 | 5 | -5 |

**Jogos:**

- 13/06/2026 — Brazil 0–0 Morocco
- 13/06/2026 — Haiti 0–1 Scotland
- 19/06/2026 — Scotland 0–0 Morocco
- 19/06/2026 — Brazil 3–0 Haiti
- 24/06/2026 — Scotland 0–1 Brazil
- 24/06/2026 — Morocco 1–0 Haiti

## Grupo E

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Germany ✅ | 7 | 3 | 2 | 1 | 0 | 5 | 1 | +4 |
| 2 | Ecuador ✅ | 7 | 3 | 2 | 1 | 0 | 3 | 1 | +2 |
| 3 | Côte d'Ivoire 🟡 | 3 | 3 | 1 | 0 | 2 | 1 | 3 | -2 |
| 4 | Curaçao | 0 | 3 | 0 | 0 | 3 | 0 | 4 | -4 |

**Jogos:**

- 14/06/2026 — Germany 2–0 Curaçao
- 14/06/2026 — Côte d'Ivoire 0–1 Ecuador
- 20/06/2026 — Germany 2–0 Côte d'Ivoire
- 20/06/2026 — Ecuador 1–0 Curaçao
- 25/06/2026 — Curaçao 0–1 Côte d'Ivoire
- 25/06/2026 — Ecuador 1–1 Germany

## Grupo F

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Netherlands ✅ | 7 | 3 | 2 | 1 | 0 | 4 | 1 | +3 |
| 2 | Sweden ✅ | 7 | 3 | 2 | 1 | 0 | 3 | 1 | +2 |
| 3 | Tunisia 🟡 | 1 | 3 | 0 | 1 | 2 | 0 | 2 | -2 |
| 4 | Japan | 1 | 3 | 0 | 1 | 2 | 0 | 3 | -3 |

**Jogos:**

- 14/06/2026 — Netherlands 2–0 Japan
- 14/06/2026 — Sweden 1–0 Tunisia
- 20/06/2026 — Netherlands 1–1 Sweden
- 20/06/2026 — Tunisia 0–0 Japan
- 25/06/2026 — Japan 0–1 Sweden
- 25/06/2026 — Tunisia 0–1 Netherlands

## Grupo G

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Belgium ✅ | 7 | 3 | 2 | 1 | 0 | 5 | 1 | +4 |
| 2 | Iran ✅ | 5 | 3 | 1 | 2 | 0 | 4 | 2 | +2 |
| 3 | Egypt 🟡 | 2 | 3 | 0 | 2 | 1 | 1 | 3 | -2 |
| 4 | New Zealand | 1 | 3 | 0 | 1 | 2 | 0 | 4 | -4 |

**Jogos:**

- 15/06/2026 — Belgium 2–0 Egypt
- 15/06/2026 — Iran 2–0 New Zealand
- 21/06/2026 — Belgium 1–1 Iran
- 21/06/2026 — New Zealand 0–0 Egypt
- 26/06/2026 — Egypt 1–1 Iran
- 26/06/2026 — New Zealand 0–2 Belgium

## Grupo H

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Spain ✅ | 9 | 3 | 3 | 0 | 0 | 5 | 0 | +5 |
| 2 | Uruguay ✅ | 6 | 3 | 2 | 0 | 1 | 2 | 1 | +1 |
| 3 | Cabo Verde 🟡 | 1 | 3 | 0 | 1 | 2 | 0 | 3 | -3 |
| 4 | Saudi Arabia | 1 | 3 | 0 | 1 | 2 | 0 | 3 | -3 |

**Jogos:**

- 15/06/2026 — Spain 2–0 Cabo Verde
- 15/06/2026 — Saudi Arabia 0–1 Uruguay
- 21/06/2026 — Spain 2–0 Saudi Arabia
- 21/06/2026 — Uruguay 1–0 Cabo Verde
- 26/06/2026 — Cabo Verde 0–0 Saudi Arabia
- 26/06/2026 — Uruguay 0–1 Spain

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo I

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | France ✅ | 9 | 3 | 3 | 0 | 0 | 5 | 0 | +5 |
| 2 | Norway ✅ | 4 | 3 | 1 | 1 | 1 | 1 | 1 | +0 |
| 3 | Senegal 🟡 | 4 | 3 | 1 | 1 | 1 | 1 | 1 | +0 |
| 4 | Iraq | 0 | 3 | 0 | 0 | 3 | 0 | 5 | -5 |

**Jogos:**

- 16/06/2026 — France 1–0 Senegal
- 16/06/2026 — Iraq 0–1 Norway
- 22/06/2026 — France 3–0 Iraq
- 22/06/2026 — Norway 0–0 Senegal
- 26/06/2026 — Norway 0–1 France
- 26/06/2026 — Senegal 1–0 Iraq

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo J

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Argentina ✅ | 9 | 3 | 3 | 0 | 0 | 6 | 0 | +6 |
| 2 | Algeria ✅ | 4 | 3 | 1 | 1 | 1 | 2 | 3 | -1 |
| 3 | Austria 🟡 | 4 | 3 | 1 | 1 | 1 | 2 | 3 | -1 |
| 4 | Jordan | 0 | 3 | 0 | 0 | 3 | 0 | 4 | -4 |

**Jogos:**

- 16/06/2026 — Argentina 2–0 Algeria
- 16/06/2026 — Austria 1–0 Jordan
- 22/06/2026 — Argentina 2–0 Austria
- 22/06/2026 — Jordan 0–1 Algeria
- 27/06/2026 — Algeria 1–1 Austria
- 27/06/2026 — Jordan 0–2 Argentina

> ⚠️ Há seleções empatadas em P/SG/GP — desempate por confronto direto/alfabético.

## Grupo K

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Portugal ✅ | 9 | 3 | 3 | 0 | 0 | 6 | 0 | +6 |
| 2 | Colombia ✅ | 6 | 3 | 2 | 0 | 1 | 3 | 1 | +2 |
| 3 | Congo DR 🟡 | 3 | 3 | 1 | 0 | 2 | 1 | 3 | -2 |
| 4 | Uzbekistan | 0 | 3 | 0 | 0 | 3 | 0 | 6 | -6 |

**Jogos:**

- 17/06/2026 — Portugal 2–0 Congo DR
- 17/06/2026 — Uzbekistan 0–2 Colombia
- 23/06/2026 — Portugal 3–0 Uzbekistan
- 23/06/2026 — Colombia 1–0 Congo DR
- 27/06/2026 — Colombia 0–1 Portugal
- 27/06/2026 — Congo DR 1–0 Uzbekistan

## Grupo L

| Pos | Seleção | P | J | V | E | D | GP | GC | SG |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | England ✅ | 9 | 3 | 3 | 0 | 0 | 5 | 0 | +5 |
| 2 | Croatia ✅ | 6 | 3 | 2 | 0 | 1 | 3 | 1 | +2 |
| 3 | Ghana 🟡 | 3 | 3 | 1 | 0 | 2 | 1 | 2 | -1 |
| 4 | Panama | 0 | 3 | 0 | 0 | 3 | 0 | 6 | -6 |

**Jogos:**

- 17/06/2026 — England 1–0 Croatia
- 17/06/2026 — Ghana 1–0 Panama
- 23/06/2026 — England 1–0 Ghana
- 23/06/2026 — Panama 0–2 Croatia
- 27/06/2026 — Panama 0–3 England
- 27/06/2026 — Croatia 1–0 Ghana

## Resumo — 1º e 2º de cada grupo

| Grupo | 1º colocado | 2º colocado |
|:---:|:---|:---|
| A | Czech Republic | South Korea |
| B | Switzerland | Bosnia and Herzegovina |
| C | Turkey | Paraguay |
| D | Brazil | Morocco |
| E | Germany | Ecuador |
| F | Netherlands | Sweden |
| G | Belgium | Iran |
| H | Spain | Uruguay |
| I | France | Norway |
| J | Argentina | Algeria |
| K | Portugal | Colombia |
| L | England | Croatia |

> **Formato 2026:** avançam os **2 primeiros de cada grupo** + os **8 melhores terceiros** (32 seleções na fase eliminatória). Os 3º colocados (🟡) dependem da comparação entre grupos — não resolvida aqui (requer ranking dos terceiros).

---

# Mata-mata previsto (Round of 32 → Final)

Confrontos decididos pelo **Dixon-Coles de produção** (`predict_match`, campo neutro). O vencedor de cada chave é a seleção com maior probabilidade de vitória (1X2); quando o placar mais provável é empate, a chave é decidida nos **pênaltis** (vai quem tem maior probabilidade de vitória). Seleções fora do ajuste do DC são decididas pelo desempenho na fase de grupos.

## Ranking dos 3º colocados (8 melhores avançam)

| Rank | Grupo | Seleção | P | SG | GP | Classificado? |
|---:|:---:|:---|---:|---:|---:|:---:|
| 1 | C | USA | 5 | +1 | 2 | ✅ |
| 2 | D | Scotland | 4 | +0 | 1 | ✅ |
| 3 | I | Senegal | 4 | +0 | 1 | ✅ |
| 4 | J | Austria | 4 | -1 | 2 | ✅ |
| 5 | L | Ghana | 3 | -1 | 1 | ✅ |
| 6 | K | Congo DR | 3 | -2 | 1 | ✅ |
| 7 | E | Côte d'Ivoire | 3 | -2 | 1 | ✅ |
| 8 | B | Canada | 2 | -1 | 1 | ✅ |
| 9 | A | Mexico | 2 | -1 | 1 | ❌ |
| 10 | G | Egypt | 2 | -2 | 1 | ❌ |
| 11 | F | Tunisia | 1 | -2 | 0 | ❌ |
| 12 | H | Cabo Verde | 1 | -3 | 0 | ❌ |

> Critério: pontos → saldo (SG) → gols pró (GP) → ordem alfabética. Os 8 primeiros completam as 32 vagas (24 dos 1º/2º + 8 terceiros).

## Cabeças-de-chave (seeding)

> **Metodologia (transparente):** como o calendário não traz a tabela oficial de cruzamentos da FIFA, sementeei as 32 seleções por **tier** (1º > 2º > 3º) e, dentro do tier, por desempenho (P → SG → GP), montando uma chave 1×32, 2×31, … padrão. Não é o slotting oficial da FIFA; é um chaveamento reprodutível para projetar o modelo.

Seeds 1–12: vencedores de grupo · 13–24: vice · 25–32: melhores terceiros.

### Round of 32 (16 jogos)

- **Argentina** vs Canada → 1–0 **Argentina**
- **Croatia** vs Uruguay → 1–1 (Croatia nos pênaltis) **Croatia**
- **Germany** vs USA → 2–0 **Germany**
- **Brazil** vs Bosnia and Herzegovina → 2–0 **Brazil**
- **France** vs Ghana → 1–0 **France**
- **Ecuador** vs Morocco → 0–0 (Morocco nos pênaltis) **Morocco**
- **Spain** vs Austria → 2–0 **Spain**
- **Turkey** vs Norway → 1–1 (Norway nos pênaltis) **Norway**
- **Portugal** vs Côte d'Ivoire → 1–0 **Portugal**
- **Colombia** vs Iran → (decidido pelo desempenho na fase de grupos — Iran sem ajuste DC) **Colombia**
- **Belgium** vs Scotland → 1–0 **Belgium**
- **Netherlands** vs South Korea → (decidido pelo desempenho na fase de grupos — South Korea sem ajuste DC) **Netherlands**
- **England** vs Congo DR → 2–0 **England**
- **Sweden** vs Paraguay → 1–0 **Sweden**
- **Switzerland** vs Senegal → 1–0 **Switzerland**
- **Czech Republic** vs Algeria → 1–1 (Czech Republic nos pênaltis) **Czech Republic**

### Oitavas de final

- **Argentina** vs Croatia → 1–0 **Argentina**
- **Germany** vs Brazil → 0–1 **Brazil**
- **France** vs Morocco → 0–0 (France nos pênaltis) **France**
- **Spain** vs Norway → 1–0 **Spain**
- **Portugal** vs Colombia → 1–0 **Portugal**
- **Belgium** vs Netherlands → 1–1 (Belgium nos pênaltis) **Belgium**
- **England** vs Sweden → 1–0 **England**
- **Switzerland** vs Czech Republic → 1–0 **Switzerland**

### Quartas de final

- **Argentina** vs Brazil → 0–0 (Brazil nos pênaltis) **Brazil**
- **France** vs Spain → 1–1 (Spain nos pênaltis) **Spain**
- **Portugal** vs Belgium → 1–1 (Portugal nos pênaltis) **Portugal**
- **England** vs Switzerland → 0–0 (England nos pênaltis) **England**

### Semifinais

- **Brazil** vs Spain → 0–0 (Brazil nos pênaltis) **Brazil**
- **Portugal** vs England → 1–1 (England nos pênaltis) **England**

### Final

- 🏆 **Brazil** vs England → 0–0 (Brazil nos pênaltis)

### Disputa de 3º lugar

- 🥉 **Spain** vs Portugal → 1–1 (Spain nos pênaltis) **Spain**

## 🏆 Campeão previsto: **Brazil**

| Posição | Seleção |
|:---|:---|
| 🥇 Campeão | Brazil |
| 🥈 Vice | England |
| 🥉 Terceiro | Spain |

