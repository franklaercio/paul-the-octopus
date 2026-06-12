# Paul the Octopus

Pipeline de ciência de dados e machine learning para prever os resultados da Copa do Mundo
FIFA 2026. O projeto combina histórico de partidas, ranking FIFA, calibração de probabilidades
e um modelo de gols para gerar previsões reproduzíveis das 72 partidas da fase de grupos.

![Paul the Octopus](img/paul.png)

> [!IMPORTANT]
> `data/results/predictions_submission.csv` é um artefato gerado. A execução completa do pipeline
> sobrescreve esse arquivo. Revise o diff das previsões antes de fazer commit.

## Modelo atual

- **Resultado 1X2:** ensemble do classificador P2 com calibração isotônica e Dixon-Coles.
- **Placar:** matriz Dixon-Coles em campo neutro, com escala de gols `1.35`.
- **Empates competitivos:** margem flexível `DRAW_MARGIN_PROD=0.08`.
- **Validação:** splits temporais e hold-out da Copa de 2022, preservados contra vazamento.

As decisões experimentais e métricas estão documentadas em
[docs/AVALIACAO-PREVISOES-2026.md](docs/AVALIACAO-PREVISOES-2026.md).

## Início rápido

Requer Python 3.10 ou superior.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m pytest
python -m scripts.run_pipeline
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m pytest
python -m scripts.run_pipeline
```

## Estrutura

```text
data/raw/                 CSVs de entrada
data/results/             Previsões geradas
notebooks/paultheoctopus.ipynb  Pipeline principal de análise, treino e inferência
scripts/                  Validação dos dados e execução automatizada
tests/                    Testes dos contratos de entrada
docs/                     Decisões, avaliações e documentação técnica
artifacts/                Notebook executado (gerado, não versionado)
```

Para instalar apenas o ambiente de execução, sem as ferramentas de desenvolvimento:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

Validar os CSVs de entrada:

```bash
python -m scripts.validate_data
```

Executar todo o notebook e validar as previsões:

```bash
python -m scripts.run_pipeline
```

O comando gera:

| Arquivo | Descrição |
|---|---|
| `data/results/predictions_submission.csv` | Saída direta do modelo, com seleções e placar previsto |
| `artifacts/paultheoctopus.executed.ipynb` | Notebook executado, métricas e diagnósticos da execução |

O arquivo `data/results/schedule_with_predictions.csv` mantém o calendário enriquecido com
placar e resultado previsto para consumo humano. Ele deve permanecer coerente com a submissão
quando ambos representarem a mesma execução do modelo.

Para trabalhar interativamente:

```bash
jupyter lab notebooks/paultheoctopus.ipynb
```

## Qualidade

```bash
python -m ruff check scripts tests
python -m pytest
```

A workflow em `.github/workflows/pipeline.yml` instala o ambiente, valida os dados, executa os
testes e roda o notebook completo em pushes e pull requests.

## Dados

As entradas ficam em `data/raw/` e os resultados em `data/results/`. O pipeline usa CSVs locais e
não depende de GCP ou banco de dados.

| Entrada | Finalidade |
|---|---|
| `historical-results.csv` | Histórico de partidas usado no treino e nas features temporais |
| `ranking.csv` | Histórico do ranking FIFA disponível antes de cada partida |
| `matches-schedule.csv` | Calendário das 72 partidas previstas para 2026 |

O calendário usa data e horário de Brasília: `date` em `DD/MM/AAAA`, `time_brasilia` em `HH:MM`
e `timezone=GMT-3`.

## Reprodutibilidade

Alterações científicas devem ser feitas em `notebooks/paultheoctopus.ipynb`. Os scripts apenas
orquestram a execução e validam os contratos; eles não duplicam a lógica do modelo.

Antes de aceitar novas previsões:

1. Execute `python -m scripts.validate_data`.
2. Execute `python -m pytest`.
3. Execute `python -m scripts.run_pipeline`.
4. Revise as métricas do notebook executado e o diff de `data/results/`.

Consulte [docs/PIPELINE.md](docs/PIPELINE.md) para contratos, etapas e solução de problemas.

## Licença

MIT. Autor: Frank Laércio.
