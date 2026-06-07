# Plano de notificações via WhatsApp com Twilio

## Objetivo

Enviar a previsão de cada partida da Copa do Mundo 2026 pelo WhatsApp uma hora antes do início do
jogo, usando os horários de Brasília (`GMT-3`) e os resultados gerados pelo pipeline de ML.

## Limitação de grupos

A API oficial do WhatsApp usada pela Twilio não permite enviar mensagens diretamente para um
grupo nativo do WhatsApp. Cada mensagem precisa ter um número individual como destinatário, no
formato `whatsapp:+<numero E.164>`.

A solução adotada será representar o grupo como uma lista de participantes e enviar a mesma
notificação individualmente para cada pessoa habilitada. A Twilio Conversations oferece um grupo
virtual, mas ele não aparece como um grupo nativo no WhatsApp e depende de janelas de atendimento
iniciadas pelos participantes.

Referências:

- [Twilio WhatsApp Group Messaging](https://www.twilio.com/code-exchange/whatsapp-group-messaging)
- [Twilio Messages API](https://www.twilio.com/docs/messaging/api/message-resource)
- [Twilio WhatsApp Quickstart](https://www.twilio.com/docs/whatsapp/quickstart)

## Fontes de dados

### Calendário

`data/raw/matches-schedule.csv` fornece:

- número da partida;
- data no horário de Brasília;
- horário em `time_brasilia`;
- seleções participantes.

### Previsões

`data/results/predictions_submission.csv` fornece:

- seleção considerada mandante;
- placar previsto;
- seleção considerada visitante.

Os arquivos serão associados pela ordem da partida. Como melhoria posterior, o arquivo de
previsões poderá incluir explicitamente a coluna `match` para eliminar a dependência da posição.

## Participantes

Criar `data/whatsapp-recipients.csv`:

```csv
name,phone,enabled
Frank,+5511999999999,true
Maria,+5511888888888,true
```

Regras:

- números no padrão internacional E.164;
- somente participantes com consentimento explícito;
- `enabled=false` interrompe os envios sem remover o registro;
- o arquivo real com números pessoais não deve ser público.

Caso o repositório permaneça público, versionar apenas
`data/whatsapp-recipients.example.csv` e ignorar o arquivo real no Git.

## Modelo da mensagem

Criar e aprovar um Content Template no console da Twilio:

```text
Jogo em 1 hora!

{{1}} x {{2}}
Horário: {{3}} (Brasília)
Previsão: {{1}} {{4}} x {{5}} {{2}}
```

Variáveis:

| Variável | Conteúdo |
|---|---|
| `1` | Primeira seleção |
| `2` | Segunda seleção |
| `3` | Horário de Brasília |
| `4` | Gols previstos da primeira seleção |
| `5` | Gols previstos da segunda seleção |

Mensagens iniciadas pelo sistema fora da janela de atendimento do WhatsApp devem utilizar um
template previamente aprovado.

## Componentes

### `scripts/build_notification_queue.py`

Responsável por:

1. validar calendário e previsões;
2. combinar cada partida com seu placar previsto;
3. criar o instante do jogo com timezone `America/Sao_Paulo`;
4. calcular `send_at = kickoff_at - 1 hora`;
5. produzir a fila de notificações pendentes.

### `scripts/send_twilio_whatsapp.py`

Cliente isolado da Twilio, responsável por:

- carregar credenciais do ambiente;
- enviar o Content Template;
- receber e retornar o `MessageSid`;
- tratar erros de autenticação, validação e indisponibilidade;
- oferecer modo de simulação sem chamada externa.

### `scripts/notification_worker.py`

Worker executado periodicamente, responsável por:

1. buscar partidas cujo `send_at` esteja na janela atual;
2. carregar participantes habilitados;
3. verificar se a mensagem já foi enviada;
4. enviar uma mensagem individual para cada participante;
5. registrar sucesso ou falha;
6. repetir somente falhas temporárias.

## Configuração

Variáveis de ambiente:

```env
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_CONTENT_SID=
```

Regras de segurança:

- nunca versionar `.env` ou tokens;
- usar secrets da plataforma de hospedagem;
- preferir API Key restrita em produção quando suportado;
- trocar imediatamente qualquer credencial exposta.

Adicionar o SDK ao `requirements.txt`:

```text
twilio>=9,<10
```

## Persistência e idempotência

Usar SQLite em `state/notifications.sqlite3` com os campos:

| Campo | Descrição |
|---|---|
| `match_id` | Número da partida |
| `recipient` | Número do participante |
| `scheduled_at` | Momento planejado do envio |
| `sent_at` | Momento efetivo do envio |
| `message_sid` | Identificador retornado pela Twilio |
| `status` | `pending`, `sent`, `delivered`, `failed` |
| `attempts` | Quantidade de tentativas |
| `last_error` | Último erro recebido |

Criar uma restrição única para `(match_id, recipient)`. Essa restrição é a proteção principal
contra mensagens duplicadas em reinicializações ou execuções concorrentes.

## Agendamento

Executar o worker a cada cinco minutos. Uma notificação fica elegível quando seu horário de envio
está dentro de uma janela controlada, por exemplo:

```text
agora - 2 minutos <= send_at < agora + 5 minutos
```

O worker deve sempre usar `America/Sao_Paulo`, e não o timezone configurado no servidor.

Opções de hospedagem:

1. servidor Linux com `cron` ou timer do systemd;
2. serviço de container com scheduler, como Cloud Run Jobs;
3. função agendada em uma plataforma serverless.

GitHub Actions não é recomendado como agendador principal porque execuções com `schedule` podem
atrasar. Pode ser mantido apenas como fallback ou execução manual.

## Retentativas

- repetir apenas falhas temporárias e respostas de limite de taxa;
- usar backoff exponencial;
- limitar a três tentativas;
- não tentar enviar depois do início da partida;
- não repetir mensagens marcadas como `sent` ou `delivered`.

## Monitoramento

Configurar um status callback da Twilio para atualizar os estados:

- `queued`;
- `sent`;
- `delivered`;
- `read`;
- `failed`;
- `undelivered`.

Os logs não devem registrar tokens nem números completos. Números podem ser mascarados, mantendo
somente os quatro últimos dígitos.

## Interface de linha de comando

Comandos planejados:

```bash
python -m scripts.build_notification_queue
python -m scripts.notification_worker --dry-run
python -m scripts.notification_worker --once
```

O modo `--dry-run` deve mostrar destinatário mascarado, partida, horário e mensagem sem chamar a
Twilio.

## Testes

Adicionar:

```text
tests/test_notification_queue.py
tests/test_twilio_sender.py
tests/test_notification_worker.py
```

Cenários mínimos:

- associação correta entre calendário e previsão;
- cálculo exato de uma hora antes em `America/Sao_Paulo`;
- mudança correta da data em jogos depois da meia-noite;
- rejeição de números fora do padrão E.164;
- exclusão de participantes desabilitados;
- envio único por partida e participante;
- nenhuma chamada externa no modo `--dry-run`;
- retentativa de falhas temporárias;
- ausência de retentativa para erros permanentes;
- bloqueio de envio depois do início da partida.

Os testes do cliente devem simular a Twilio; a CI não deve enviar mensagens reais.

## Etapas de implementação

1. Criar o arquivo de exemplo dos participantes e regras de privacidade.
2. Adicionar o SDK da Twilio e configuração por variáveis de ambiente.
3. Implementar e testar a associação calendário/previsões.
4. Implementar a fila com timezone e horário de envio.
5. Implementar o cliente Twilio com `--dry-run`.
6. Criar e aprovar o Content Template no console da Twilio.
7. Implementar SQLite, idempotência e retentativas.
8. Implementar o worker e status callback.
9. Configurar o scheduler e os secrets no ambiente de produção.
10. Fazer teste real com um único número autorizado.
11. Habilitar gradualmente os demais participantes.

## Critérios de aceite

- cada participante habilitado recebe uma única mensagem por partida;
- a tentativa ocorre uma hora antes, com tolerância máxima de cinco minutos;
- data e horário exibidos correspondem a Brasília;
- mensagem contém as seleções e o placar previsto correto;
- nenhuma credencial ou número pessoal é versionado;
- falhas ficam registradas e podem ser auditadas;
- o pipeline de ML continua funcionando sem depender da Twilio;
- a CI executa todos os testes sem enviar mensagens reais.
