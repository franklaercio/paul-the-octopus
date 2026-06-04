# Paul the Octopus — Contexto do Projeto

## Visão Geral

Paul the Octopus é um projeto de ciência de dados e machine learning cujo objetivo é prever os resultados de partidas da Copa do Mundo FIFA utilizando dados históricos de jogos internacionais e o ranking oficial da FIFA. O nome é uma referência ao famoso polvo Paul, que ficou mundialmente conhecido por suas "previsões" durante a Copa do Mundo de 2010.

O pipeline completo cobre coleta de dados, análise exploratória, engenharia de features, treinamento de modelo e geração de predições — tudo em um único notebook interativo.

---

## Linguagem e Ambiente

| Item | Detalhe |
|---|---|
| Linguagem principal | Python 3 |
| Ambiente de execução | Jupyter Notebook / Google Colab |
| Licença | MIT |
| Autor | Frank Laércio |

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e análise de dados tabulares |
| `numpy` | Operações de álgebra linear e arrays |
| `matplotlib` | Visualizações gráficas |
| `seaborn` | Visualizações estatísticas avançadas |
| `scikit-learn` | Algoritmos de machine learning e métricas |
| `google-cloud / gsutil` | Integração com Google Cloud Storage |

Não há arquivo `requirements.txt` ou `pyproject.toml` — as dependências estão implícitas no notebook e no ambiente Colab.

---

## Estrutura do Projeto

```
paul-the-octopus/
├── files/
│   ├── historical-results.csv                  # Histórico de partidas (44.061 registros)
│   ├── ranking.csv                             # Ranking FIFA histórico (63.917 registros)
│   ├── matches-schedule.csv                    # Calendário de jogos a prever (48 partidas)
│   ├── historical_win-loose-draw_ratios.csv    # Estatísticas head-to-head (799 registros)
│   ├── predictions_submission.csv              # Predições geradas pelo modelo
│   └── shootouts.csv                           # Dados de disputas por pênaltis (504 registros)
├── img/
│   └── paul.png                                # Banner do projeto
├── src/
│   └── paultheoctopus.ipynb                    # Notebook principal
├── context.md                                  # Este arquivo
├── README.md
└── LICENSE
```

---

## Base de Dados / Fontes de Dados

O projeto não utiliza banco de dados relacional ou NoSQL. Toda a persistência é feita em arquivos CSV armazenados localmente e/ou no Google Cloud Storage.

### `historical-results.csv`
Histórico completo de partidas internacionais de futebol desde 1872.

| Campo | Descrição |
|---|---|
| `date` | Data da partida |
| `home_team` | Time mandante |
| `away_team` | Time visitante |
| `home_score` | Gols do mandante |
| `away_score` | Gols do visitante |
| `tournament` | Competição (Copa, Eliminatórias, etc.) |
| `city` | Cidade sede |
| `country` | País sede |
| `neutral` | Se o jogo foi em campo neutro |

### `ranking.csv`
Histórico do ranking FIFA desde 1992.

| Campo | Descrição |
|---|---|
| `rank` | Posição no ranking |
| `country_full` | Nome completo do país |
| `country_abrv` | Código de 3 letras |
| `total_points` | Pontuação FIFA |
| `previous_points` | Pontuação anterior |
| `rank_change` | Variação de posição |
| `confederation` | Confederação (UEFA, CONMEBOL, CAF, AFC…) |
| `rank_date` | Data de referência do ranking |

### `matches-schedule.csv`
Calendário das partidas da Copa do Mundo a serem previstas.

| Campo | Descrição |
|---|---|
| `match` | Número da partida |
| `date` | Data |
| `country1` | Seleção 1 |
| `country2` | Seleção 2 |
| `phase` | Fase da competição |

### `historical_win-loose-draw_ratios.csv`
Estatísticas históricas de confrontos diretos (head-to-head) entre pares de seleções.

| Campo | Descrição |
|---|---|
| `country1` / `country2` | Par de seleções |
| `games` | Total de jogos disputados |
| `wins` | Taxa de vitórias do country1 |
| `looses` | Taxa de derrotas do country1 |
| `draws` | Taxa de empates |

### `shootouts.csv`
Resultados de disputas por pênaltis em jogos oficiais.

| Campo | Descrição |
|---|---|
| `date` | Data da partida |
| `home_team` / `away_team` | Times |
| `winner` | Vencedor nos pênaltis |

---

## Pipeline de Machine Learning

### 1. Preparação dos Dados

- Partidas anteriores a 2000 são descartadas (baixa relevância histórica)
- Filtragem de torneios: apenas os 12 mais competitivos são mantidos:
  - FIFA World Cup, Copa América, UEFA Nations League
  - Eliminatórias FIFA, Eliminatórias UEFA Euro
  - African Cup of Nations, AFC Asian Cup
  - CECAFA Cup, CFU Caribbean Cup qualification, Gulf Cup, entre outros
- Amistosos, British Championship e Merdeka Tournament são removidos
- Torneios são codificados como IDs numéricos (0–11)

### 2. Engenharia de Features

| Feature | Cálculo | Significado |
|---|---|---|
| `rank_difference` | rank_mandante − rank_visitante | Distância relativa no ranking |
| `average_rank` | (rank_mandante + rank_visitante) / 2 | Nível médio do confronto |
| `score_difference` | gols_mandante − gols_visitante | Resultado em gols |
| `is_won` | score_difference > 0 | Variável alvo (vitória do mandante) |

### 3. Modelo

| Parâmetro | Valor |
|---|---|
| Algoritmo | Random Forest Classifier |
| Estimadores | 100 árvores |
| Features de entrada | `average_rank`, `rank_difference` |
| Variável alvo | `is_won` (binário) |
| Acurácia | **94,24%** |

### 4. Lógica de Predição de Placar

Com base na probabilidade de vitória calculada pelo modelo:

| Probabilidade de vitória | Gols atribuídos |
|---|---|
| < 60% | 0 gols |
| 60% – 70% | 1 gol |
| > 70% | 2 gols |

A probabilidade do time visitante é calculada como `1 - P(mandante)`.

---

## Análise de Dados (EDA)

- Análise de frequência de torneios (gráficos de barras)
- Detecção e tratamento de valores nulos
- Filtragem temporal (pós-2000)
- Análise de correlação entre diferença de ranking e resultado
- Distribuição histórica de vitórias, empates e derrotas por seleção

---

## Conexões e Integrações Externas

### Google Cloud Platform (GCP)

| Item | Valor |
|---|---|
| Projeto | `phoenix-cit` |
| Bucket GCS | `gs://paul-the-octopus-frank-junior/` |
| Autenticação | `google.colab.auth.authenticate_user()` |
| Ferramenta | `gsutil` CLI |

**Fluxo de dados:**
1. Download dos CSVs do GCS para o ambiente Colab no início da execução
2. Upload do arquivo `predictions_submission.csv` de volta ao GCS após geração

---

## Branches

| Branch | Propósito |
|---|---|
| `main` | Versão estável / produção |
| `feature/2026-world-cup` | Desenvolvimento das predições para a Copa 2026 |

---

## Limitações Conhecidas

- Não há arquivo de dependências (`requirements.txt`), dificultando reprodução fora do Colab
- Configurações de GCP (project ID, bucket) estão hardcoded no notebook
- Não há testes automatizados
- O modelo utiliza apenas 2 features (ranking), sem considerar forma recente, histórico de confrontos, baixas ou local do jogo
- A lógica de atribuição de placar é heurística, não estatística
