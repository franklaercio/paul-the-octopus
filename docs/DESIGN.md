# Design Document — Paul the Octopus

## 1. Objetivo

Construir um sistema de predição de resultados da Copa do Mundo FIFA baseado em dados históricos de partidas internacionais e no ranking oficial da FIFA, utilizando um modelo de machine learning capaz de classificar o resultado (vitória/derrota) do time mandante para cada confronto da competição.

---

## 2. Contexto e Motivação

Prever resultados de partidas de futebol é um problema de classificação binária com alta variância intrínseca. A abordagem deste projeto parte de uma hipótese direta: **a diferença de ranking FIFA entre dois times é o principal preditor do resultado de um confronto**. Dados históricos desde 2000 são utilizados para treinar e validar essa hipótese.

O nome "Paul the Octopus" é uma referência cultural ao polvo Paul, que ficou famoso durante a Copa do Mundo de 2010 por acertar os resultados das partidas da Alemanha e da final do torneio.

---

## 3. Arquitetura Geral

O sistema segue uma arquitetura linear de pipeline de dados, sem serviços distribuídos ou APIs. Todo o processamento ocorre em um único notebook Jupyter executado no Google Colab.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Cloud Storage                     │
│              gs://paul-the-octopus-frank-junior/                │
└────────────────────────┬────────────────────────────────────────┘
                         │ gsutil cp (download)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Google Colab / Jupyter                       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  Data Ingest │──▶│     EDA      │──▶│ Feature          │   │
│  │  & Cleaning  │   │  & Filtering │   │ Engineering      │   │
│  └──────────────┘   └──────────────┘   └────────┬─────────┘   │
│                                                  │             │
│                                         ┌────────▼─────────┐   │
│                                         │  Model Training  │   │
│                                         │  (Random Forest) │   │
│                                         └────────┬─────────┘   │
│                                                  │             │
│                                         ┌────────▼─────────┐   │
│                                         │   Prediction     │   │
│                                         │   Generation     │   │
│                                         └────────┬─────────┘   │
└──────────────────────────────────────────────────┼─────────────┘
                         │ gsutil cp (upload)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               predictions_submission.csv (GCS)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Decisões de Design

### 4.1 Escolha do Algoritmo — Random Forest

**Decisão:** Utilizar `RandomForestClassifier` com 100 estimadores como modelo principal.

**Justificativa:**
- Robusto a overfitting em datasets de tamanho moderado
- Lida bem com features não-lineares sem necessidade de normalização
- Fornece probabilidades de classe (`predict_proba`), necessárias para a lógica de placar
- Baixo custo de hiperparametrização para um baseline sólido

**Alternativas descartadas:**
- Regressão logística: menos expressiva para padrões não-lineares
- Redes neurais: custo de implementação desproporcional ao volume de dados
- SVM: não fornece probabilidades nativamente sem calibração adicional

---

### 4.2 Janela Temporal — Dados a partir de 2000

**Decisão:** Descartar partidas anteriores ao ano 2000.

**Justificativa:**
- O futebol moderno (formações, preparação física, análise tática) difere substancialmente do futebol pré-2000
- O ranking FIFA só foi introduzido em 1992 e levou anos para se estabilizar como métrica confiável
- Reduz ruído nos dados de treino sem perda significativa de volume

---

### 4.3 Filtragem de Torneios

**Decisão:** Manter apenas os 12 torneios mais competitivos e remover amistosos.

**Justificativa:**
- Amistosos têm baixa intensidade competitiva — times frequentemente poupam titulares
- Torneios regionais menores (British Championship, Merdeka Tournament) têm contexto cultural específico e pouca representatividade global
- Torneios classificatórios e principais competições têm maior correlação entre ranking e desempenho real

**Torneios mantidos:**
| ID | Torneio |
|---|---|
| 0 | FIFA World Cup qualification |
| 1 | UEFA Euro qualification |
| 2 | African Cup of Nations qualification |
| 3 | FIFA World Cup |
| 4 | Copa América |
| 5 | AFC Asian Cup qualification |
| 6 | African Cup of Nations |
| 7 | CECAFA Cup |
| 8 | CFU Caribbean Cup qualification |
| 9 | UEFA Nations League |
| 10 | Gulf Cup |
| 11 | AFC Asian Cup |

---

### 4.4 Features de Entrada

**Decisão:** Utilizar apenas `average_rank` e `rank_difference` como features.

**Justificativa:**
- Simplicidade e interpretabilidade — o ranking FIFA já é uma métrica composta que agrega forma, resultados e força de oponentes
- Evita vazamento de dados (data leakage) ao usar informações que estariam disponíveis antes da partida
- Duas features bem escolhidas superam muitas features ruidosas

**Features descartadas nesta versão:**
- Histórico head-to-head (disponível no CSV, mas não integrado ao modelo)
- Forma recente (últimos N jogos)
- Local da Copa (fator campo neutro)
- Pênaltis históricos

---

### 4.5 Lógica de Placar — Regra Heurística

**Decisão:** Converter probabilidade de vitória em placar via faixas fixas.

**Justificativa:**
- Predição exata de placar é um problema de muito maior complexidade (regressão de contagem com distribuição de Poisson)
- A abordagem heurística mantém o escopo do projeto focado na classificação
- Os limiares (60%/70%) foram definidos para gerar resultados plausíveis e diferenciar os níveis de favoritismo

```
P(vitória mandante) < 60%  →  0 gols para o mandante
P(vitória mandante) 60–70% →  1 gol  para o mandante
P(vitória mandante) > 70%  →  2 gols para o mandante

P(vitória visitante) = 1 - P(vitória mandante)
```

---

### 4.6 Armazenamento — CSV em vez de Banco de Dados

**Decisão:** Usar arquivos CSV como única camada de persistência.

**Justificativa:**
- Volume de dados compatível com CSV (< 65k linhas por arquivo)
- Simplicidade de reprodução — não requer instalação de servidor de banco de dados
- Integração direta com pandas sem camada de ORM ou driver adicional
- Dados são essencialmente imutáveis (histórico de partidas não muda)

---

## 5. Fluxo de Dados Detalhado

```
historical-results.csv ──┐
ranking.csv ─────────────┤
                         ▼
                  [ Merge por data e país ]
                         │
                         ▼
                  [ Filtro: ano >= 2000 ]
                         │
                         ▼
                  [ Filtro: torneios selecionados ]
                         │
                         ▼
                  [ Feature Engineering ]
                  rank_difference, average_rank,
                  score_difference, is_won
                         │
                         ▼
                  [ Treino: RandomForestClassifier ]
                  X = [average_rank, rank_difference]
                  y = is_won
                         │
                         ▼
                  [ Acurácia: 94.24% ]
                         │
         ┌───────────────┘
         │
matches-schedule.csv ────┐
ranking.csv (atual) ─────┤
                         ▼
                  [ Predição por partida ]
                  predict_proba → P(home_win)
                         │
                         ▼
                  [ Atribuição de placar ]
                  via regras heurísticas
                         │
                         ▼
              predictions_submission.csv
```

---

## 6. Estrutura de Arquivos de Dados

| Arquivo | Linhas | Período | Atualização |
|---|---|---|---|
| `historical-results.csv` | ~44.061 | 1872–atual | Batch (antes do torneio) |
| `ranking.csv` | ~63.917 | 1992–atual | Batch (antes do torneio) |
| `matches-schedule.csv` | 48 | Copa atual | Manual (calendário da FIFA) |
| `historical_win-loose-draw_ratios.csv` | ~799 | Histórico | Batch |
| `shootouts.csv` | ~504 | Histórico | Batch |
| `predictions_submission.csv` | 49 | Gerado | Output do modelo |

---

## 7. Limitações e Dívidas Técnicas

| Limitação | Impacto | Possível solução |
|---|---|---|
| Apenas 2 features no modelo | Baixa capacidade preditiva para upset | Adicionar forma recente, head-to-head |
| Lógica de placar heurística | Placares pouco realistas | Modelo de regressão de Poisson |
| Configurações hardcoded no notebook | Dificulta manutenção | Arquivo de configuração separado |
| Sem `requirements.txt` | Reprodução fora do Colab instável | Adicionar `requirements.txt` ou `pyproject.toml` |
| Sem testes automatizados | Regressões silenciosas | Adicionar testes com pytest |
| Autenticação manual no Colab | Não automatizável em CI/CD | Service account key |
| Dados desatualizados no CSV | Predições imprecisas | Pipeline de atualização automática |

---

## 8. Evolução Planejada — Copa 2026

Para a edição de 2026, os seguintes aprimoramentos estão planejados na branch `feature/2026-world-cup`:

- [ ] Atualizar `historical-results.csv` com resultados de 2022–2026
- [ ] Atualizar `ranking.csv` com o ranking FIFA mais recente
- [ ] Atualizar `matches-schedule.csv` com o calendário oficial da Copa 2026
- [ ] Adicionar feature de forma recente (últimos 10 jogos)
- [ ] Integrar dados de confrontos diretos (`historical_win-loose-draw_ratios.csv`) ao modelo
- [ ] Avaliar modelo de Poisson para predição de placar
- [ ] Adicionar `requirements.txt`
- [ ] Separar configurações em arquivo `.env` ou `config.yaml`
