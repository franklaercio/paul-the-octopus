# AGENTS.md

Guia de contexto para o Codex trabalhar neste repositório. Para a documentação completa, veja [CONTEXT.md](CONTEXT.md).

## O que é

**Paul the Octopus** — projeto de ciência de dados / ML que prevê resultados da Copa do Mundo FIFA a partir de dados históricos de partidas internacionais e do ranking oficial da FIFA. Todo o pipeline (coleta → EDA → features → treino → predição) vive em um único notebook.

## Stack e ambiente

- **Python 3**, executado em **Jupyter Notebook / Google Colab**
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `google-cloud`/`gsutil`
- **Não há** `requirements.txt` nem `pyproject.toml` — dependências são implícitas no ambiente Colab
- Licença MIT — Autor: Frank Laércio

## Estrutura

```
files/    CSVs de dados (histórico, ranking, calendário, head-to-head, pênaltis) + predições geradas
img/      banner do projeto
src/      paultheoctopus.ipynb  ← notebook principal (todo o código vive aqui)
docs/     DESIGN.md, VALIDACAO-2022.md
```

## Pipeline de ML (resumo)

1. **Preparação:** descarta partidas anteriores a 2000; mantém só os 12 torneios mais competitivos; remove amistosos; torneios viram IDs numéricos (0–11).
2. **Features:** `rank_difference` (rank mandante − visitante), `average_rank` ((mandante+visitante)/2), `score_difference`, alvo `is_won` (score_difference > 0).
3. **Modelo:** Random Forest Classifier, 100 árvores. Entradas: `average_rank`, `rank_difference`. Acurácia reportada: **94,24%**.
4. **Placar (heurística):** P(vitória) < 60% → 0 gols; 60–70% → 1 gol; > 70% → 2 gols. Visitante = `1 − P(mandante)`.

## Integração externa (GCP)

- Projeto `phoenix-cit`, bucket `gs://paul-the-octopus-frank-junior/`
- Autenticação via `google.colab.auth.authenticate_user()`, transferências com `gsutil`
- Fluxo: baixa os CSVs do GCS no início → gera `predictions_submission.csv` → faz upload de volta
- ⚠️ Project ID e bucket estão **hardcoded** no notebook

## Branches

- `main` — versão estável
- `feature/2026-world-cup` — desenvolvimento das predições da Copa 2026 (branch atual)

## Convenções e armadilhas

- **Todo o código está no notebook** `src/paultheoctopus.ipynb` — não há módulos `.py`. Edições de código acontecem nas células do notebook.
- **Dados são CSVs** em `files/` — sem banco de dados.
- O modelo usa **apenas 2 features** (ranking); não considera forma recente, head-to-head, baixas ou mando de campo.
- A atribuição de placar é **heurística, não estatística**.
- **Sem testes automatizados** — validação manual (ver `docs/VALIDACAO-2022.md`).
- Documentação e comentários do projeto são majoritariamente em **português**.
