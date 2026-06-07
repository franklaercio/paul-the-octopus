# Paul the Octopus

Pipeline de ciência de dados e machine learning para prever os resultados da Copa do Mundo
FIFA 2026 a partir de partidas internacionais, ranking FIFA, forma recente e modelos 1X2/placar.

![Paul the Octopus](img/paul.png)

## Estrutura

```text
files/                    CSVs de entrada e previsões geradas
src/paultheoctopus.ipynb  Pipeline principal de análise, treino e inferência
scripts/                  Validação dos dados e execução automatizada
tests/                    Testes dos contratos de entrada
docs/                     Decisões, avaliações e documentação técnica
artifacts/                Notebook executado (gerado, não versionado)
```

## Requisitos

- Python 3.10 ou superior
- Ambiente virtual recomendado

```bash
python -m venv .venv
```

Ativação no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

Instale o ambiente de execução:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para desenvolvimento e CI:

```bash
python -m pip install -r requirements-dev.txt
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

- `files/predictions_submission.csv`: previsões do modelo;
- `artifacts/paultheoctopus.executed.ipynb`: notebook executado com os resultados.

Para trabalhar interativamente:

```bash
jupyter lab src/paultheoctopus.ipynb
```

## Qualidade

```bash
python -m ruff check scripts tests
python -m pytest
```

A workflow em `.github/workflows/pipeline.yml` instala o ambiente, valida os dados, executa os
testes e roda o notebook completo em pushes e pull requests.

## Dados

Os CSVs ficam em `files/`. O pipeline não depende de GCP ou banco de dados. O calendário de
2026 usa data e horário de Brasília (`GMT-3`) nas colunas `date`, `time_brasilia` e `timezone`.

Consulte [docs/PIPELINE.md](docs/PIPELINE.md) para contratos, etapas e solução de problemas.

## Licença

MIT. Autor: Frank Laércio.
