# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

**Paul the Octopus** é um pipeline de ciência de dados e machine learning em Python para prever as **72 partidas da fase de grupos da Copa do Mundo FIFA 2026** (calendário em `data/raw/matches-schedule.csv`). O nome é uma referência ao polvo Paul, que "previa" jogos de Copa.

O projeto está em **recomeço**: a estrutura do pipeline existe como esqueleto, mas as decisões de modelagem (alvo, algoritmo, features, validação, métricas) são **intencionalmente deixadas em aberto**, para serem tomadas ao longo do tempo. Não assuma um modelo específico.

Autor: Frank Laércio. Licença MIT.

## Stack e ambiente

- **Python 3.10+** (o CI roda em 3.11).
- Núcleo de dados/ML: **numpy, pandas, scipy, scikit-learn**.
- Visualização: **matplotlib, seaborn**.
- Notebooks: **jupyterlab, ipykernel, nbclient, nbformat**; artefatos intermediários em **pyarrow** (Parquet).
- Qualidade (dev): **pytest, ruff** (em `requirements-dev.txt`).
- Persistência em **CSV/Parquet**; sem banco de dados e sem dependência de GCP.

`requirements.txt` é o ambiente de execução; `requirements-dev.txt` adiciona teste/lint.

## Arquitetura

Regra central: **a lógica científica vive nos notebooks**; os scripts em `scripts/` apenas **orquestram e validam** — não duplicam a ciência.

O pipeline é dividido em notebooks por etapa, quebrados nas fronteiras de artefato (cada um lê o que o anterior gravou):

```text
data/raw/*.csv
    │
    ▼
scripts/validate_data.py        ← valida colunas obrigatórias e o contrato do calendário
    │
    ▼
notebooks/01_features.ipynb     → data/processed/features.parquet
    │
    ▼
notebooks/02_train.ipynb        → models/model.joblib                         (quando implementado)
    │
    ▼
notebooks/03_predict.ipynb      → data/results/predictions_submission.csv     (quando implementado)

notebooks/00_eda.ipynb          ← exploração, FORA do pipeline executável
artifacts/*.executed.ipynb      ← notebooks executados (não versionado)
```

- Hoje os notebooks são **esqueletos**: `01` grava um `features.parquet` de partida (pass-through do histórico, com `# TODO` para as features); `02` e `03` carregam suas entradas e deixam treino/inferência como `# TODO`.
- `scripts/validate_data.py` define os schemas em `REQUIRED_COLUMNS` e valida o calendário (numeração única e sequencial, data `DD/MM/AAAA`, hora `HH:MM`, `timezone=GMT-3`). Falha **antes** de rodar o pipeline quando um contrato é violado.
- `scripts/run_pipeline.py` valida as entradas, executa `01 → 02 → 03` via `nbclient` (`allow_errors=False`) e grava os notebooks executados em `artifacts/`. Conforme cada etapa for implementada, **acrescente a validação do artefato de saída** (features, modelo, submission) no script.
- Evite **vazamento temporal**: cada feature deve usar apenas informação conhecida antes da partida.

## Comandos

```bash
python -m pip install -r requirements-dev.txt   # ambiente completo (execução + dev)
python -m scripts.validate_data                 # valida os contratos dos CSVs de entrada
python -m ruff check scripts                    # lint
python -m scripts.run_pipeline                  # executa 01 → 02 → 03 e valida as entradas
```

Trabalhar interativamente num notebook:

```bash
jupyter lab notebooks/01_features.ipynb
```

`run_pipeline` aceita `--timeout <segundos>` (por célula, padrão 900) e `--output-dir <caminho>`.

## Dados (`data/raw/`)

| Arquivo | Uso |
|---|---|
| `historical-results.csv` | Histórico de partidas internacionais |
| `ranking.csv` | Histórico do ranking FIFA |
| `matches-schedule.csv` | Calendário das 72 partidas de grupos de 2026 (horário de Brasília) |
| `shootouts.csv` | Histórico de disputas de pênaltis (auxiliar) |
| `historical_win-loose-draw_ratios.csv` | Razões de confronto direto (auxiliar) |
| `worldcup-2026-results.csv` | Resultados reais de 2026 conforme acontecem (comparação) |

O validador cobre os três primeiros. O calendário usa `date` em `DD/MM/AAAA`, `time_brasilia` em `HH:MM` e `timezone=GMT-3` (o validador **exige** `GMT-3`).

Artefatos gerados ficam em `data/processed/` (features, Parquet) e `models/` (modelos): o git versiona só a pasta (`.gitkeep`), não o conteúdo.

## Convenções

- A ciência vive **nos notebooks**; `scripts/` só orquestra/valida.
- `00_eda.ipynb` é exploração e **não entra** no `run_pipeline`.
- Mudou o schema de um CSV? Atualize `scripts/validate_data.py` (`REQUIRED_COLUMNS`) **e** os testes em `tests/`.
- Não versione `artifacts/`, `data/processed/`, `models/`, ambientes virtuais nem checkpoints.
- Documentação e comentários são majoritariamente em **português** (o `README.md` é a exceção, em inglês).

## Agentes (`.claude/agents/`)

Três subagentes especializados (todos `model: opus`, em português) — consulte-os ao tomar as decisões de modelagem:

- **`football-data-scientist`** — perfil completo: decisões de modelagem, escolha de algoritmos e métricas, revisão crítica do pipeline. Coordena os outros dois.
- **`python-ml-engineer`** — engenharia em Python: pandas/numpy vetorizado, pipelines scikit-learn sem vazamento, reprodutibilidade, `ruff`/`pytest`.
- **`football-analyst`** — domínio do futebol: leitura tática e fatores de contexto (desfalques, fadiga, mando, clima, calendário), atualizado via busca na web.

`.claude/settings.json` libera comandos Python e edição de notebook, e pede confirmação para remoções (`rm`, `git rm`, `git clean`) e `git push`.

## Estado do repositório

A branch `feature/worldcup2026-new-agents-ai` está em reorganização (recomeço do pipeline):

- O pipeline atual é o esqueleto de notebooks acima. O **modelo anterior** (notebook único `notebooks/paultheoctopus.ipynb`) e a documentação antiga (`docs/`, `CONTEXT.md`, `README.md`) seguem no **histórico do git** / `main` para referência (ex.: `git show main:notebooks/paultheoctopus.ipynb`), mas **não** ditam as decisões deste recomeço.
- Arquivo rastreado ainda ausente na árvore de trabalho: `tests/test_validate_data.py` (contrato de dados — reconstrua ao retomar os testes; o CI roda `ruff check scripts tests` e `pytest`). `settings.json` referencia um caminho legado `src/paultheoctopus.ipynb`.
