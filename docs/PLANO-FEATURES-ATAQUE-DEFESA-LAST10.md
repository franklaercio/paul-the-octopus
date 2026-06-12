# Plano — Features de Ataque e Defesa Last10

## Objetivo

Adicionar features pre-jogo baseadas nos ultimos 10 jogos de cada selecao,
medindo gols marcados e sofridos recentes, e avaliar se elas melhoram o modelo
sem vazamento de informacao futura.

O foco principal e melhorar a geracao de placares. Para 1X2, ranking/Elo ja
carregam muito sinal; para placar, separar perfil ofensivo e defensivo pode
ajudar a distinguir vitorias como `1-0`, `2-0`, `2-1` ou `3-1`.

## Escopo

Implementar no notebook principal:

```text
src/paultheoctopus.ipynb
```

Preferencialmente dentro da secao P4 de engenharia de features, reaproveitando
`build_football_features` e os testes anti-vazamento ja existentes.

## Features Propostas

Criar as seguintes features:

```text
gf_last10_home
gf_last10_away
ga_last10_home
ga_last10_away
attack_diff_last10 = gf_last10_home - gf_last10_away
defense_diff_last10 = ga_last10_home - ga_last10_away
home_pressure_last10 = gf_last10_home - ga_last10_away
away_pressure_last10 = gf_last10_away - ga_last10_home
pressure_diff_last10 = home_pressure_last10 - away_pressure_last10
```

Definicoes:

- `gf_last10`: media de gols marcados pela selecao nos 10 jogos anteriores.
- `ga_last10`: media de gols sofridos pela selecao nos 10 jogos anteriores.
- `attack_diff_last10`: vantagem ofensiva recente do mandante/calendario.
- `defense_diff_last10`: diferenca de gols sofridos recentes.
- `pressure_diff_last10`: diferenca entre pressao ofensiva esperada do mandante
  e do visitante.

## Regras de Implementacao

1. Usar somente jogos anteriores a partida atual.
2. Aplicar `shift(1)` antes do rolling.
3. Considerar jogos da selecao como mandante e visitante juntos.
4. Usar janela movel de 10 jogos.
5. Nao usar `home_score` nem `away_score` da propria partida como feature dela
   mesma.
6. Preencher valores ausentes com mediana do treino ou media global historica,
   nunca com o placar da propria partida.
7. Na inferencia da Copa 2026, as linhas futuras com placar placeholder nao
   podem afetar as proprias features.

## Plano Tecnico

### 1. Criar visao longa por selecao

Atualizar `team_long_view(hist)` ou criar helper novo:

```python
def team_long_view_goals(hist):
    ...
```

Cada partida deve virar duas linhas:

```text
mi, date, team, opponent, gf, ga
```

Para o mandante:

```text
team = home_team
opponent = away_team
gf = home_score
ga = away_score
```

Para o visitante:

```text
team = away_team
opponent = home_team
gf = away_score
ga = home_score
```

### 2. Calcular rolling last10 sem vazamento

Usar `shift(1)` antes da janela movel:

```python
long['gf_last10'] = (
    long.groupby('team')['gf']
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)

long['ga_last10'] = (
    long.groupby('team')['ga']
        .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)
```

`min_periods=3` evita estatisticas muito instaveis no inicio da serie. Valores
ausentes restantes devem ser imputados no pipeline, como ja acontece no P4.

### 3. Mapear de volta para o dataframe de partidas

Criar colunas no historico:

```python
hist['gf_last10_home']
hist['gf_last10_away']
hist['ga_last10_home']
hist['ga_last10_away']
```

Usar o indice original da partida (`mi`) e o lado (`home`/`away`) para garantir
alinhamento correto.

### 4. Criar diferencas e pressao

```python
hist['attack_diff_last10'] = hist['gf_last10_home'] - hist['gf_last10_away']
hist['defense_diff_last10'] = hist['ga_last10_home'] - hist['ga_last10_away']

hist['home_pressure_last10'] = hist['gf_last10_home'] - hist['ga_last10_away']
hist['away_pressure_last10'] = hist['gf_last10_away'] - hist['ga_last10_home']
hist['pressure_diff_last10'] = (
    hist['home_pressure_last10'] - hist['away_pressure_last10']
)
```

## Validacao Anti-Vazamento

Adicionar teste especifico para as novas features:

1. Escolher uma partida `t`.
2. Guardar as features last10 dessa partida.
3. Alterar artificialmente o placar da propria partida para `9-0`.
4. Recalcular as features.
5. Confirmar que as features last10 da partida `t` nao mudaram.

Exemplo:

```python
before = df_feat.loc[t, LAST10_FEATURES].copy()

df_fake = df_model.copy()
df_fake.loc[t, 'home_score'] = 9
df_fake.loc[t, 'away_score'] = 0
df_fake_feat, _ = build_football_features(df_fake)

after = df_fake_feat.loc[t, LAST10_FEATURES]
assert before.equals(after)
```

Tambem validar que o primeiro jogo de uma selecao nao usa o placar do proprio
jogo para preencher `gf_last10` ou `ga_last10`.

## Avaliacao

Adicionar as novas features aos candidatos do P4:

```python
P4_CANDIDATES += [
    'attack_diff_last10',
    'defense_diff_last10',
    'pressure_diff_last10',
]
```

Opcionalmente testar componentes brutas:

```python
[
    'gf_last10_home',
    'gf_last10_away',
    'ga_last10_home',
    'ga_last10_away',
]
```

Recomendacao inicial: comecar pelas diferenciais para reduzir overfit.

## Criterio de Aceitacao

Adotar as features apenas se melhorarem fora da amostra.

Minimo:

```text
RPS_holdout(P4 + last10) <= RPS_holdout(modelo atual)
```

Ideal:

```text
RPS melhora
Brier nao piora
log-loss nao piora
acerto 1X2 nao cai muito
pontuacao de placar 10/5/0 nao piora
```

Se melhorar validacao mas piorar hold-out, manter o experimento documentado como
rejeitado, seguindo o padrao atual do notebook.

## Entregaveis

1. Notebook atualizado com secao P4.x ou P7.x para features last10.
2. Relatorio impresso no notebook com:
   - ablacao individual das novas features;
   - features aceitas/rejeitadas;
   - comparacao P2/P3/P4 atual vs. P4+last10.
3. CSVs regenerados apenas se as features forem adotadas no modelo de producao.
4. Nota curta em `docs/AVALIACAO-PREVISOES-2026.md` explicando o resultado do
   experimento.

## Commit Sugerido

```text
Add last10 attack defense feature experiment
```
