# Pipeline

## Visão geral

O notebook `notebooks/paultheoctopus.ipynb` continua sendo a implementação principal. Os scripts não
duplicam a lógica do modelo: eles validam os contratos dos dados, executam o notebook e verificam
se a saída esperada foi produzida.

```text
data/raw/*.csv
    |
    v
scripts.validate_data
    |
    v
notebooks/paultheoctopus.ipynb
    |
    +--> data/results/predictions_submission.csv
    +--> artifacts/paultheoctopus.executed.ipynb
```

## Entradas obrigatórias

### `historical-results.csv`

Histórico de partidas usado para treino e engenharia de features. O validador exige as colunas
`date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country` e
`neutral`.

### `ranking.csv`

Histórico do ranking FIFA. O validador exige as colunas `rank`, `country_full`, `country_abrv`,
`total_points`, `previous_points`, `rank_change`, `confederation` e `rank_date`.

### `matches-schedule.csv`

Calendário de inferência. Os números de partida devem ser únicos e sequenciais. Datas usam
`DD/MM/AAAA`, horários usam `HH:MM` e o fuso deve ser `GMT-3`.

| Coluna | Descrição |
|---|---|
| `match` | Número sequencial da partida |
| `date` | Data no horário de Brasília |
| `time_brasilia` | Horário de Brasília |
| `timezone` | Valor fixo `GMT-3` |
| `country1` | Primeira seleção |
| `country2` | Segunda seleção |
| `phase` | Fase da competição |

## Comandos

### Validar entradas

```bash
python -m scripts.validate_data
```

O comando falha antes do treino quando um arquivo está ausente, uma coluna obrigatória foi
removida ou o calendário contém data, horário, fuso ou numeração inválidos.

### Executar o pipeline

```bash
python -m scripts.run_pipeline
```

Opções disponíveis:

```bash
python -m scripts.run_pipeline --timeout 1200 --output artifacts/execucao.ipynb
```

O timeout é aplicado por célula. Após a execução, o script exige que
`predictions_submission.csv` tenha as colunas de placar e a mesma quantidade de partidas do
calendário.

## Integração contínua

A workflow `Pipeline` executa em pushes para `main` e `feature/2026-world-cup`, em pull requests
e manualmente. O notebook executado é publicado como artefato da execução do GitHub Actions.

## Atualização de dados

1. Atualize os CSVs em `data/raw/` sem alterar seus contratos.
2. Execute `python -m scripts.validate_data`.
3. Execute `python -m scripts.run_pipeline`.
4. Revise as métricas do notebook e `data/results/predictions_submission.csv` antes do commit.
