# Design do projeto

## Objetivo

Prever resultados da Copa do Mundo FIFA 2026 usando dados históricos de seleções e ranking FIFA,
com validação temporal e geração de probabilidades 1X2 e placares.

## Arquitetura

O projeto usa uma arquitetura batch local. Não há banco de dados, API ou dependência obrigatória
de serviços de nuvem.

```text
CSVs locais
   |
   v
Validação de contratos
   |
   v
Notebook Jupyter
   |
   +--> preparação e EDA
   +--> engenharia de features
   +--> treino e validação temporal
   +--> inferência da Copa 2026
   |
   v
CSV de previsões + notebook executado
```

## Decisões

- O notebook permanece como fonte principal para preservar a narrativa analítica do projeto.
- Scripts externos cuidam apenas de orquestração e validação, sem duplicar código do modelo.
- Dados e resultados tabulares usam CSV pelo volume reduzido e facilidade de reprodução.
- O calendário usa data e horário de Brasília (`GMT-3`) para consumo no Brasil.
- O ambiente é declarado por `requirements.txt`; ferramentas de CI ficam em
  `requirements-dev.txt`.
- A CI executa contratos, testes e o notebook completo para detectar regressões de integração.

## Limitações

- A execução completa ainda é monolítica porque o código científico vive em um notebook.
- Atualizações dos CSVs são manuais.
- Resultados dependem da data do ranking e do histórico versionados no repositório.
- Extração futura de funções estáveis para módulos Python deve ocorrer apenas quando houver testes
  de equivalência suficientes para evitar mudança silenciosa nas previsões.

Veja [PIPELINE.md](PIPELINE.md) para os contratos e comandos operacionais.
