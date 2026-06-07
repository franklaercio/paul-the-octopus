# Paul the Octopus - Contexto do projeto

## Visão geral

Projeto de ciência de dados e machine learning para prever partidas da Copa do Mundo FIFA 2026.
O pipeline completo vive em `src/paultheoctopus.ipynb` e cobre preparação dos dados, EDA,
engenharia de features, validação temporal, treino, calibração, placares e inferência.

## Ambiente

- Python 3.10+
- Jupyter Notebook / JupyterLab / Google Colab
- Dependências de execução em `requirements.txt`
- Dependências de teste e lint em `requirements-dev.txt`
- Dados locais em CSV; não há dependência obrigatória de GCP ou banco de dados

Principais bibliotecas: pandas, NumPy, SciPy, Matplotlib, Seaborn e scikit-learn.

## Estrutura

```text
files/                    Dados e previsões em CSV
src/paultheoctopus.ipynb  Implementação principal do pipeline
scripts/validate_data.py  Validação dos contratos de entrada
scripts/run_pipeline.py   Execução automatizada e validação da saída
tests/                    Testes dos contratos de dados
docs/                     Design, avaliações e planos técnicos
.github/workflows/        Integração contínua
artifacts/                Notebook executado, não versionado
```

## Dados principais

- `historical-results.csv`: histórico de partidas internacionais.
- `ranking.csv`: histórico do ranking FIFA.
- `matches-schedule.csv`: 72 partidas da fase de grupos de 2026, em horário de Brasília.
- `predictions_submission.csv`: saída direta do notebook.
- `schedule_with_predictions.csv`: calendário enriquecido com placares e resultados previstos.
- `historical_win-loose-draw_ratios.csv` e `shootouts.csv`: bases auxiliares.

O calendário usa `date` em `DD/MM/AAAA`, `time_brasilia` em `HH:MM` e `timezone=GMT-3`.

## Pipeline de ML

1. Carrega dados locais via `DATA_DIR`, aceitando execução na raiz ou em `src/`.
2. Descarta partidas anteriores a 2000 e filtra competições relevantes.
3. Junta resultados históricos ao ranking FIFA disponível antes de cada partida.
4. Cria features de ranking, Elo e forma recente sem usar informações futuras.
5. Faz splits temporais para validação e hold-out de 2022.
6. Compara modelos 1X2 e calibração de probabilidades.
7. Usa Dixon-Coles para probabilidades de gols e placares.
8. Gera as previsões das 72 partidas de grupos da Copa 2026.

O notebook é a fonte principal do modelo. Scripts externos devem orquestrar e validar, sem
duplicar a lógica científica.

## Operação

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.validate_data
python -m pytest
python -m scripts.run_pipeline
```

O último comando gera `artifacts/paultheoctopus.executed.ipynb` e valida se
`files/predictions_submission.csv` contém o mesmo número de partidas do calendário.

## Convenções

- Documentação e comentários são majoritariamente em português.
- Edições da lógica do modelo acontecem nas células do notebook.
- Alterações de contratos dos CSVs exigem atualização do validador e dos testes.
- Não versionar ambientes virtuais, checkpoints ou `artifacts/`.
- Preservar splits temporais e guardas contra vazamento de dados.

## Documentação relacionada

- `docs/PIPELINE.md`: execução e contratos operacionais.
- `docs/DESIGN.md`: decisões de arquitetura.
- `docs/VALIDACAO-2022.md`: validação manual histórica.
- `docs/AVALIACAO-PREVISOES-2026.md`: avaliação dos modelos atuais.
- `docs/PLANO-MELHORIAS.md`: histórico e planejamento técnico.
