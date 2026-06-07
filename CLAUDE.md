# CLAUDE.md

Guia de contexto para o Claude trabalhar neste repositório. Para detalhes, veja
[CONTEXT.md](CONTEXT.md) e [docs/PIPELINE.md](docs/PIPELINE.md).

## Projeto

**Paul the Octopus** é um pipeline de ciência de dados e ML para prever partidas da Copa do
Mundo FIFA 2026. A implementação científica principal vive em `notebooks/paultheoctopus.ipynb`.

## Ambiente

- Python 3.10+
- Dependências: `requirements.txt`
- Desenvolvimento e CI: `requirements-dev.txt`
- Entradas em `data/raw/` e resultados em `data/results/`; não há dependência obrigatória de GCP
- Persistência em CSV; não há banco de dados

## Estrutura

```text
data/raw/                 Entradas em CSV
data/results/             Previsões geradas
notebooks/paultheoctopus.ipynb  Pipeline principal
scripts/                  Validação e execução automatizada
tests/                    Testes dos contratos de dados
docs/                     Design, avaliações e planos
.github/workflows/        Pipeline de CI
```

## Comandos

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m ruff check scripts tests
python -m pytest
python -m scripts.run_pipeline
```

## Convenções

- Alterações do modelo acontecem nas células do notebook.
- Scripts externos orquestram e validam; não devem duplicar a lógica científica.
- Mudanças nos schemas dos CSVs exigem atualização de `scripts/validate_data.py` e dos testes.
- O calendário usa data e horário de Brasília: `time_brasilia` e `timezone=GMT-3`.
- Preserve os splits temporais e as guardas contra vazamento de dados.
- Não versione `artifacts/`, ambientes virtuais ou checkpoints.
- Documentação e comentários são majoritariamente em português.
