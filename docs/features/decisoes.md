# Decisões de Engenharia de Features — `01_features.ipynb`

Especificação **fechada** que o engenheiro de ML (`python-ml-engineer`) deve seguir
ao implementar o `01`. Resolve as "Decisões em aberto" (§8 do
[`README.md`](README.md)) com fórmulas e valores exatos. Cada decisão traz uma
justificativa breve ancorada na EDA e nos dados crus (verificados ao escrever esta
spec).

**Regras invioláveis (valem para tudo abaixo):**

- **Zero vazamento temporal.** Toda feature usa apenas informação conhecida
  **antes do apito inicial** da partida. Elo registra o valor pré-jogo e atualiza
  depois; janelas móveis usam `shift(1)` na série de cada seleção; ranking via
  `merge_asof` **estritamente antes** da data.
- **Nunca ler `data/raw/worldcup-2026-results.csv`.** É o gabarito.
- **Consistência treino↔previsão.** A *mesma* função computa as features para o
  histórico (`split="train"`) e para os 72 jogos de 2026 (`split="predict"`).
- **Agnóstico ao modelo.** Gravar valores **crus** (sem escalonar/one-hot — isso é
  do `02`); gravar alvo 1X2 **e** placares.
- **Reprodutível.** Semente fixa (`SEED = 42`), operações determinísticas e
  vetorizadas.
- **Fonte única de nomes.** `canon()` a partir do `DEFAULT_ALIASES` de
  `.claude/skills/avaliar-previsoes/scripts/score_predictions.py` (importar/replicar
  literalmente; não criar outro mapa). Verificado: as 48 seleções de 2026 resolvem
  1:1 e todas existem no histórico **e** no ranking.

---

## 0. Limpeza de entrada (E0) — pré-requisitos das demais decisões

1. **Dedupe do histórico.** A chave `(date, home_team, away_team)` tem **1 par
   duplicado com placares divergentes**: `1974-02-17 Tahiti x New Caledonia`
   aparece como `2x1` e `1x2`. Não é linha idêntica — é conflito. **Decisão:**
   remover **ambas** as linhas desse par (`keep=False` na chave). Justificativa: 2
   linhas em 49.402 é irrelevante para o sinal, e manter um placar arbitrário
   injeta ruído; descartar é o mais conservador e determinístico.
2. **Sem placares nulos.** Verificado: `home_score`/`away_score` não têm NaN. Nada
   a tratar.
3. **Tabela longa por seleção-jogo.** Construir a partir do histórico já
   canônico/deduplicado uma tabela com **uma linha por (seleção, partida)** em
   ordem cronológica, com colunas: `date`, `team`, `opponent`, `gf` (gols pró),
   `ga` (gols contra), `is_home` (bool), `is_neutral`, `points` (3/1/0),
   `tournament`. É a base de Elo, forma e descanso. Empates de data resolvem-se
   por uma ordenação estável `(date, home_team, away_team)`.

---

## 1. Elo (E1)

Modelo de força contínua, calculado **varrendo todo o histórico** em ordem
cronológica (de 1872 até 2026-06-10). Para cada partida: ler o Elo pré-jogo dos
dois times (vira feature), computar o resultado esperado, atualizar depois.

**Parâmetros (fixos):**

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Rating inicial | `1500` | Convenção World Football Elo; centro neutro. |
| `K` (fator de atualização) | `40` | Padrão consagrado do eloratings.net para seleções; reativo o suficiente sem oscilar demais com ~poucos jogos/ano por seleção. |
| `HFA` (vantagem de mando) | `+65` no Elo do mandante, **só quando `is_neutral == False`** | Aproxima o efeito de mando observado (não-neutro 0,507 vs neutro 0,442 de vitória do mandante). Aplicado **apenas no cálculo do esperado**, não persiste no rating. |
| Multiplicador de margem (G) | **Sim** (fórmula abaixo) | Vencer por goleada é evidência mais forte; reduz ruído de vitórias magras. Padrão eloratings.net. |
| Estreantes | Entram com `1500`; sem cold-start especial | Verificado: **todas as 48 seleções de 2026 já têm histórico abundante** (mín. 38 jogos desde 2022). O "estreante" só existe lá no passado do varrimento e auto-regula com K=40. |

**Fórmulas (exatas):**

Resultado esperado do mandante (com `dr` = diferença de Elo já incluindo HFA):

```
dr      = (elo_home + (0 if is_neutral else HFA)) - elo_away
E_home  = 1 / (1 + 10 ** (-dr / 400))
E_away  = 1 - E_home
```

Resultado real `S_home ∈ {1, 0.5, 0}` (vitória/empate/derrota do mandante);
`S_away = 1 - S_home`.

Multiplicador de margem de gols (`gd` = |home_score − away_score|):

```
G = 1                         se gd <= 1
G = 1.5                       se gd == 2
G = (11 + gd) / 8            se gd >= 3
```

Atualização (após o jogo):

```
elo_home_novo = elo_home + K * G * (S_home - E_home)
elo_away_novo = elo_away + K * G * (S_away - E_away)
```

**Features gravadas:** `elo_home`, `elo_away` (pré-jogo) e
`elo_diff = elo_home - elo_away` (**sem** HFA — a informação de mando vai separada
em `is_neutral`, para o modelo aprender o peso). O `expected_home` (E_home) **não**
é gravado como feature (é função determinística do diff+HFA; evita redundância).

**Atenção do engenheiro:** o Elo dos 72 jogos de 2026 é o **estado final do
varrimento** (último Elo conhecido de cada seleção até 2026-06-10). Como na fase de
grupos cada seleção só joga os 3 jogos do grupo e **não realimentamos** resultados
do Mundial, o Elo de uma seleção é **o mesmo nos seus 3 jogos** de 2026 (snapshot
pré-torneio). Isso é correto e esperado — não tente atualizar Elo entre rodadas de
2026 (seria vazar resultado).

---

## 2. Ranking FIFA *as-of* (E1)

Juntar ao jogo o **último ranking publicado estritamente antes da data** da
partida, via `merge_asof(direction="backward", allow_exact_matches=False)` por
seleção (após `canon()`).

**Decisões:**

1. **Usar `rank` E `total_points`** como features cruas, mas a feature de força
   principal é **`points_diff`**, não o nível absoluto. Motivo (verificado): o
   `total_points` **mudou de escala drasticamente** — pré-2018 ia de ~1 a ~2000
   (sistema antigo, somatório); a partir de meados de 2018 passou ao sistema
   FIFA-Elo (~800–1900). O **nível** de pontos não é comparável entre eras; a
   **diferença** entre dois adversários medidos na mesma data é muito mais estável.
   Por isso:
   - Gravar `rank_home`, `rank_away`, `rank_diff = rank_away - rank_home`
     (positivo = mandante melhor colocado, já que rank menor é melhor),
     `points_diff = points_home - points_away`.
   - **Não** gravar `total_points` absoluto como feature (cardinalidade de escala
     enganosa); só o `points_diff`. Se quiser manter `points_home`/`points_away`
     para auditoria, deixar claro no `02` que não devem virar feature crua sem
     normalização por época.
2. **Pré-1993 = NaN.** O ranking começa em 1992-12-31. Para qualquer partida antes
   da 1ª publicação aplicável, `rank_*`/`points_diff` ficam **NaN** (não imputar no
   `01`). Justificativa: imputar inventaria sinal; o tratamento (imputação por
   época, indicador de ausência, ou modelo que aceita NaN como XGBoost/LightGBM) é
   decisão do `02`. **Todos os 72 jogos de 2026 têm ranking** (publicação de
   2026-04), então a previsão nunca fica sem ranking.
3. **21 `rank` nulos no ranking.** São linhas com `total_points` presente mas
   `rank` ausente (ex.: American Samoa, Samoa, Tonga, Eritrea em algumas
   publicações 2023–2026 — seleções de baixa atividade). **Decisão:** ao construir
   a tabela de ranking *as-of*, **descartar as linhas com `rank` nulo antes do
   `merge_asof`** (não propagar). Como `total_points` existe nessas linhas, ainda
   assim **não** as use para `points_diff` (descartá-las inteiras mantém `rank` e
   `points` coerentes na mesma publicação). Nenhuma das 48 seleções de 2026 é
   afetada na publicação que vale para a previsão.

---

## 3. Forma recente (E3)

Capta momento. Calculada na **tabela longa**, por seleção, com `shift(1)` para a
linha nunca enxergar o próprio jogo.

**Decisões:**

- **Janela `N = 5`** jogos. Justificativa: 5 é o padrão de "forma" no futebol e
  equilibra reatividade × ruído; com seleções jogando poucos jogos/ano, 5 cobre
  ~6–12 meses. Usar `min_periods=1` (computa com o que houver; primeiros jogos da
  história ficam com janela parcial, o que é honesto).
- **Métricas (por seleção, deslocadas):**
  - `form_pts_home` / `form_pts_away`: média de pontos (3/1/0) nos últimos N jogos.
  - `form_gf_home` / `form_gf_away`: média de gols **pró** nos últimos N.
  - `form_ga_home` / `form_ga_away`: média de gols **contra** nos últimos N.
  - As três últimas (gf/ga) servem **também** de força ofensiva/defensiva (base de
    um modelo de gols). Não criar features de "força" separadas — `form_gf/ga` já
    cumprem esse papel; manter o conjunto enxuto.
- **Ponderação por recência: NÃO.** Decisão: **média simples** na janela de 5.
  Justificativa: a recência já está embutida na janela curta; ponderação
  exponencial dentro de 5 jogos muda pouco e adiciona um hiperparâmetro sem ganho
  medido (a EDA não evidenciou necessidade). O decaimento por recência **existe**,
  mas no `sample_weight` (item 8), não na forma. Simplicidade justificada.
- **Computação na tabela longa, não no jogo.** Calcular as 3 médias móveis por
  `team` (com `groupby("team")` + `shift(1)` + `rolling(N).mean()`) e depois mapear
  de volta para o lado `home`/`away` de cada partida. Isso garante que a forma de um
  time considera **todos** os seus jogos (em casa e fora), não só os de mando.

---

## 4. Mando / campo neutro (E2)

A EDA mostrou que campo neutro derruba a vitória do mandante (0,507→0,442) e sobe
a do visitante (0,264→0,334). Tratar `is_neutral` corretamente é decisivo em 2026.

**Decisões:**

1. **Histórico:** `is_neutral` = a coluna `neutral` do `historical-results.csv`
   (já booleana, sem nulos; média 0,264, coerente com a EDA). Usar como está.
2. **2026 — proxy de mando (limitação registrada):** o `matches-schedule.csv`
   **não tem coluna de sede/cidade**, e o gabarito `worldcup-2026-results.csv` é
   **proibido**. Proxy adotado:

   > **Um jogo de 2026 é não-neutro (`is_neutral = False`) somente quando o `home`
   > nominal do calendário é um dos anfitriões; caso contrário é neutro
   > (`is_neutral = True`).**

   **Anfitriões (lista fechada, em forma canônica):** `usa`, `canada`, `mexico`.

   Verificado no calendário: **9 jogos** têm anfitrião como `home` (3 de cada:
   USA, Canadá, México) e **nenhum** anfitrião aparece como `away`. Logo o proxy
   gera **9 jogos não-neutros e 63 neutros**, limpo e sem ambiguidade.

   **Limitação a registrar no notebook e no contrato:** este proxy assume que
   anfitriões jogam em casa e que todos os demais jogam em sede neutra. Não é
   perfeito — na prática há cidades-sede e um time pode ter "quase-mando" por
   geografia/torcida — mas é a melhor aproximação **sem** a coluna de sede e **sem**
   tocar no gabarito. Se um CSV de sedes/cidades for adicionado depois, refinar
   aqui (ponto único de mudança).
3. **`is_neutral` é feature** (bool) e governa o HFA do Elo (item 1). Não derivar
   uma `home_advantage` numérica no `01`; o modelo aprende o peso de `is_neutral`.

---

## 5. `rest_days` (E4)

Fadiga/calendário. Calculado na **tabela longa** por seleção.

**Decisões:**

- **Definição:** `rest_days = (data do jogo) − (data do jogo anterior da mesma
  seleção)`, em dias. Para o **primeiro** jogo histórico de uma seleção (sem jogo
  anterior), `rest_days = NaN` (não imputar no `01`).
- **Cap em 30 dias:** `rest_days_capped = min(rest_days, 30)`. Justificativa:
  acima de ~1 mês a "fadiga" satura — 45 ou 400 dias de descanso são
  funcionalmente equivalentes ("vem descansado"); o cap evita que pausas longas
  (entre torneios, ou o hiato de uma seleção fraca) dominem a escala. Gravar a
  versão **capada** como feature (`rest_days_home`, `rest_days_away`).
- **2026:** calcular a partir do **próprio calendário** dos 72 jogos. Para o
  **primeiro** jogo de cada seleção no Mundial **não existe jogo anterior dentro do
  calendário** — usar como "jogo anterior" o **último jogo da seleção no histórico**
  (≤ 2026-06-10) e aplicar o mesmo cap de 30. Isso mantém a definição idêntica
  treino↔previsão (a 1ª rodada de 2026 fica com `rest_days ≈ 30` capado, o que é
  realista: as seleções chegam descansadas). **Nunca** usar resultados do Mundial,
  só **datas** do calendário — datas não vazam resultado.

---

## 6. Confronto direto — H2H (E5)

**Decisão: COMPUTAR do histórico, NÃO usar `historical_win-loose-draw_ratios.csv`.**

Justificativa (verificado):

- O arquivo `ratios` tem 798 pares dirigidos mas envolve **só 31 seleções
  distintas** e cobre apenas **14 dos 72** confrontos de 2026.
- Computar do próprio histórico (em qualquer ordem de mando) cobre **44 dos 72**
  confrontos — mais que o triplo — e usa a **fonte única** (sem um segundo arquivo
  para reconciliar/validar). Mantém uma só verdade.

**Como computar (sem vazamento):** para cada partida, o H2H deve considerar
**apenas jogos anteriores** entre as duas seleções. Implementação vetorizada
sugerida: ordenar o histórico por data, agrupar por **par não-ordenado**
`frozenset({home, away})` (ou par ordenado canônico `tuple(sorted([a,b]))`), e
acumular com `shift(1)` dentro do par os contadores de vitórias/empates do mandante
**daquele** confronto. Para 2026, o H2H é o **acumulado de toda a história** entre
as duas (snapshot pré-torneio).

**Features gravadas:**

- `h2h_games`: nº de confrontos prévios entre as duas (inteiro ≥ 0).
- `h2h_winrate_home`: fração de vitórias **do mandante atual** nos confrontos
  prévios, **orientada ao mando deste jogo** — i.e., vitórias do time que é `home`
  agora (independente de quem foi mandante nos jogos passados), sobre `h2h_games`.
  Empates contam como 0 na taxa de vitória (são capturados implicitamente; manter
  simples — não criar `h2h_drawrate` separado, esparso demais).
- `h2h_available`: bool, `True` sse `h2h_games > 0`.

**Valor default quando ausente** (`h2h_games == 0`, ~28/72 dos jogos de 2026):
`h2h_winrate_home = 0.5` (neutro, sem informação) e `h2h_available = False`.
Justificativa: 0.5 é o "não sei" honesto; o `h2h_available=False` permite ao `02`
tratar o caso (interação, ou ignorar a feature quando ausente). **Cautela
explícita:** H2H é esparso e ruidoso (poucos jogos, gerações diferentes de
jogadores) — é Tier 3, sinal fraco; está aqui por completude, não como aposta.

---

## 7. Alvo e orientação das linhas (E6)

**Decisões:**

1. **Manter alvo 1X2 e placares** (agnóstico ao modelo):
   - `target_outcome ∈ {"home", "draw", "away"}` — derivado do placar
     (`home_score > away_score` → `home`; `<` → `away`; `=` → `draw`). Estes
     rótulos casam com a ordem ordinal `OUTCOMES = ("home","draw","away")` da skill
     de avaliação (importa para o RPS). Só em `split="train"`.
   - `home_score`, `away_score` (inteiros) — só em `train`; sustentam um modelo de
     gols (Poisson/Dixon-Coles). A EDA confirma que vale manter essa porta aberta:
     gols ~Poisson com leve dependência em placares baixos (empate observado 0,2274
     vs 0,2349 sob independência; correlação home/away ≈ −0,145). Esse ajuste é do
     **`02`**, não do `01` — o `01` só grava os placares crus.
2. **Orientação: home-oriented (uma linha por partida), SEM aumento simétrico no
   `01`.** Justificativa: o aumento simétrico (espelhar cada jogo trocando
   home/away) é uma técnica de **treino** que ajuda a remover viés de mando quando
   há muitos jogos neutros — mas é uma decisão do **`02`** (afeta amostragem/peso,
   e o conjunto de previsão dos 72 jogos é fixo home-oriented pelo calendário).
   Fazer o espelhamento no `01` **dobraria** as linhas de treino e quebraria a
   simetria train/predict (não se espelha o predict). **Recomendação ao `02`:**
   como 63 dos 72 jogos de 2026 são neutros, **avaliar** o aumento simétrico (ou,
   melhor, garantir que o modelo trate `is_neutral` corretamente para que o mando
   não seja confundido com força). O `01` entrega as linhas home-oriented e a
   coluna `is_neutral` que torna isso possível.

---

## 8. `sample_weight` (E6)

Peso de treino = **importância do torneio × decaimento por recência**. Só em
`split="train"`. Justificativa: amistosos dominam o histórico (18.389 jogos, ~37%)
e se comportam diferente; e jogos de 1950 informam pouco sobre 2026. O peso corrige
ambos sem **descartar** dados (o Elo ainda varre tudo — ver item 9).

```
sample_weight = w_torneio * w_recencia
```

**(a) `w_torneio`** — mapa de baldes por palavra-chave (aplicar na ordem; primeira
que casar vence; case-insensitive sobre `tournament`). Cobre os tipos vistos na
EDA (200 torneios distintos):

| Ordem | Balde | Regex/palavras-chave (case-insensitive) | `w_torneio` | Ordinal de importância |
|---|---|---|---|---|
| 1 | Copa do Mundo (fase final) | `^fifa world cup$` (exato, **sem** "qualification") | `3.0` | 5 |
| 2 | Torneio continental principal + Confederations | `euro\b`, `copa am[eé]rica`, `african cup of nations`, `asian cup`, `gold cup`, `concacaf championship`, `oceania nations cup`, `confederations cup` — **excluir** os que contêm `qualification`/`qualifying` | `2.5` | 4 |
| 3 | Eliminatória de Copa | `world cup qualification` | `2.0` | 3 |
| 4 | Eliminatória continental + Nations League | `qualification`, `qualifying`, `nations league` | `1.5` | 2 |
| 5 | Amistoso | `^friendly$` | `1.0` | 1 |
| 6 | Outros (default) | qualquer outro (torneios menores, jogos regionais, CONIFA, copas antigas) | `0.8` | 0 |

Notas de implementação do mapa (validadas contra os 200 torneios da EDA):
- Casar **Copa do Mundo final** (`^fifa world cup$`) **antes** da eliminatória,
  senão "FIFA World Cup qualification" cairia no balde errado.
- "UEFA Euro qualification" e "African Cup of Nations qualification" devem cair em
  **Eliminatória continental** (balde 4), não no continental principal — por isso a
  exclusão de `qualification` no balde 2 (confirmado: Euro→2.5 e Euro qual→1.5;
  AFC Asian Cup→2.5 e qual→1.5; Gold Cup→2.5 e qual→1.5).
- **`oceania nations cup`** deve ser ancorado **completo** (não a keyword solta
  `nations cup`): há um torneio antigo genérico "Nations Cup" (6 jogos) que **não**
  é continental e deve cair em "outros". O `confederations cup` (140 jogos, torneio
  FIFA) entra no balde 2.
- Distribuição resultante sobre todo o histórico (sanity check): amistoso ~37%,
  outros ~20%, WC-qual ~18%, cont-qual/NL ~17%, continental ~7%, WC ~2%.
- O **ordinal de importância** (coluna à direita) é gravado como feature opcional
  `tournament_tier` (inteiro 0–5) — ordem **real** (amistoso < eliminatória <
  continental < Copa), útil a modelos lineares; **não** rotular os 200 torneios como
  `0..199` cru. O **one-hot** de baldes (se desejado) fica no `02`. Em 2026 todos os
  72 jogos são balde "Copa do Mundo" / `tournament_tier = 5`.

**(b) `w_recencia`** — decaimento exponencial pela idade do jogo, com **meia-vida
de 8 anos**:

```
idade_anos    = (DATA_REF - date).days / 365.25
w_recencia    = 0.5 ** (idade_anos / HALFLIFE_ANOS)     # HALFLIFE_ANOS = 8
```

onde `DATA_REF = 2026-06-11` (1º dia da Copa; constante fixa no notebook, **não**
derivada do gabarito). Justificativa da meia-vida de 8 anos (~2 ciclos de Copa): um
jogo de 8 anos atrás pesa metade de um recente; de ~24 anos atrás, ~1/8. Captura a
não-estacionariedade entre gerações sem zerar a história mais antiga (que ainda
ancora o Elo e seleções com pouca história recente). É um valor **defensável**, não
ajustado em teste — se o `02` quiser tunar a meia-vida por validação, é um único
hiperparâmetro claro.

**Faixa resultante:** `w_torneio ∈ [0.8, 3.0]`, `w_recencia ∈ (0, 1]`; produto
sempre `> 0`. Gravar `sample_weight` como `float`. (Linhas `predict` não têm peso →
deixar `NaN`.)

---

## 9. Janela de treino (E6 / item de §8 do README)

**Decisões (duas coisas distintas, não confundir):**

1. **O Elo varre TODO o histórico** (1872 → 2026-06-10), sem filtro. É obrigatório:
   o Elo de 2026 é o acumulado de toda a trajetória; cortar anos antigos quebraria a
   recursão e zeraria seleções. O varrimento é independente da janela de treino.
2. **Linhas de treino = todas as partidas com `target_outcome` derivável** (i.e.,
   todo o histórico deduplicado), **sem corte por ano**. Justificativa: não
   descartamos dados; em vez disso, **o `sample_weight` (item 8) cuida da relevância**
   — jogos antigos e amistosos entram com peso baixo, jogos recentes e competitivos
   com peso alto. Isso é estatisticamente superior a um corte duro (que jogaria fora
   sinal de seleções com pouca história recente) e mantém o conjunto reprodutível.
   - **Features de ranking ausentes pré-1993** (NaN) **não** excluem a linha do
     treino — o tratamento de NaN é do `02` (modelos de árvore aceitam; modelos
     lineares pedem imputação/indicador). Gravar NaN e seguir.
   - **Recomendação ao `02`:** se um modelo específico (ex.: regressão logística)
     não lidar bem com a cauda antiga de baixo peso, filtrar por `sample_weight` ou
     por ano **no `02`** (ex.: treinar em `date >= 1990` ou `sample_weight > ε`) — é
     decisão de modelagem, não do `01`. O `01` entrega tudo + o peso para essa
     escolha ser feita com base medida.

---

## 10. Colunas finais e ordem do `features.parquet` (E6)

Uma linha por partida (histórico + 72 de 2026), na **ordem exata** abaixo. Tipos
conforme o contrato §2 do `README.md`, com os refinamentos desta spec.

| # | Coluna | Tipo | Aplica a | Observação |
|---|---|---|---|---|
| 1 | `split` | str | ambos | `"train"` ou `"predict"` |
| 2 | `match_no` | Int (nullable) | predict | nº do calendário; **NaN** em `train` |
| 3 | `date` | date | ambos | data da partida |
| 4 | `home` | str | ambos | canônico |
| 5 | `away` | str | ambos | canônico |
| 6 | `is_neutral` | bool | ambos | histórico = `neutral`; 2026 = proxy (item 4) |
| 7 | `confed_home` | str | ambos | confederação (do ranking *as-of*/tabela canônica) |
| 8 | `confed_away` | str | ambos | idem |
| 9 | `elo_home` | float | ambos | pré-jogo |
| 10 | `elo_away` | float | ambos | pré-jogo |
| 11 | `elo_diff` | float | ambos | `elo_home - elo_away` (sem HFA) |
| 12 | `rank_home` | float | ambos | *as-of*; NaN pré-1993 |
| 13 | `rank_away` | float | ambos | *as-of*; NaN pré-1993 |
| 14 | `rank_diff` | float | ambos | `rank_away - rank_home` (+ = mandante melhor) |
| 15 | `points_diff` | float | ambos | `points_home - points_away` (*as-of*); NaN pré-1993 |
| 16 | `form_pts_home` | float | ambos | média pts últimos 5 (shift) |
| 17 | `form_pts_away` | float | ambos | idem |
| 18 | `form_gf_home` | float | ambos | média gols pró últimos 5 (shift) |
| 19 | `form_gf_away` | float | ambos | idem |
| 20 | `form_ga_home` | float | ambos | média gols contra últimos 5 (shift) |
| 21 | `form_ga_away` | float | ambos | idem |
| 22 | `rest_days_home` | float | ambos | dias desde último jogo, **capado em 30** |
| 23 | `rest_days_away` | float | ambos | idem |
| 24 | `h2h_games` | int | ambos | confrontos prévios (computado do histórico) |
| 25 | `h2h_winrate_home` | float | ambos | vitórias do mandante atual / `h2h_games`; default `0.5` |
| 26 | `h2h_available` | bool | ambos | `h2h_games > 0` |
| 27 | `tournament_tier` | int | ambos | ordinal de importância 0–5 (item 8a); 2026 = 5 |
| 28 | `target_outcome` | str | train | `home`/`draw`/`away`; **NaN** em `predict` |
| 29 | `home_score` | Int (nullable) | train | placar; **NaN** em `predict` |
| 30 | `away_score` | Int (nullable) | train | placar; **NaN** em `predict` |
| 31 | `sample_weight` | float | train | item 8; **NaN** em `predict` |

**Notas de tipagem (Parquet):**
- Usar dtypes nullable do pandas (`Int64`, `boolean`, `string`) onde há NaN
  esperado, para o Parquet preservar o nullable sem virar `object`/`float`
  silencioso. `match_no`, `home_score`, `away_score` como `Int64`.
- `date` como `datetime64[ns]` (data pura; sem timezone — o calendário é GMT-3 mas
  só a data importa para junções *as-of*).
- Ordenar o arquivo final por `(split, date, home)` para reprodutibilidade
  determinística do artefato.

**Validações que o E7 deve garantir** (já previstas no README, reforçadas aqui):
- Exatamente **72** linhas `split == "predict"`, cobrindo `match_no` 1..72 sem
  buracos.
- `target_outcome`, `home_score`, `away_score`, `sample_weight` **presentes sse**
  `split == "train"` (e nulos em `predict`).
- Sem NaN nas chaves (`split`, `date`, `home`, `away`, `is_neutral`).
- Smoke test de vazamento: para uma partida histórica qualquer, recomputar `elo_*`
  e `form_*` "à mão" usando só jogos anteriores e conferir que batem (nenhuma
  feature enxerga o próprio jogo ou o futuro).
- **Nenhuma célula lê `worldcup-2026-results.csv`** (grep no notebook).

---

## Resumo dos valores fixos (cola rápida)

```
SEED              = 42
ELO_INICIAL       = 1500
ELO_K             = 40
ELO_HFA           = 65        # só quando is_neutral == False
# margem: G=1 (gd<=1); 1.5 (gd==2); (11+gd)/8 (gd>=3)
FORM_N            = 5         # janela; média simples; shift(1)
REST_CAP_DIAS     = 30
H2H_DEFAULT       = 0.5       # winrate quando h2h_games == 0
HALFLIFE_ANOS     = 8        # decaimento de recência no sample_weight
DATA_REF          = 2026-06-11
HOSTS             = {"usa", "canada", "mexico"}   # canônicos; 9 jogos não-neutros
# w_torneio: WC=3.0 | continental=2.5 | WC-qual=2.0 | cont-qual/NL=1.5 | amistoso=1.0 | outros=0.8
```

**Itens que esta spec deliberadamente deixa para o `02` (não implementar no `01`):**
escalonamento/one-hot (dentro do `Pipeline`), imputação de NaN, aumento simétrico,
escolha do algoritmo (classificador 1X2 vs modelo de gols/Dixon-Coles), tuning de
hiperparâmetros (incl. meia-vida do peso) e eventual filtro de janela de treino por
modelo. O `01` entrega valores crus, completos e sem vazamento — as decisões de
modelagem ficam medidas no `02`.
