# ClinicAI - Agente de Triagem para WhatsApp com LangGraph e Gemini - Seleção Decode

## Principais Funcionalidades

- **Conversa Guiada**: O agente segue um fluxo de perguntas definido para coletar dados de forma organizada (nome, idade, queixa principal, detalhes dos sintomas, etc.).
- **Extração de Dados**: Transforma a conversa em linguagem natural em um objeto JSON limpo e organizado.
- **Gerenciamento de Estado com LangGraph**: Mantém o contexto da conversa, permitindo que o agente se lembre de informações anteriores e decida os próximos passos de forma inteligente.
- **Detalhamento de Múltiplos Sintomas**: Se um paciente relata vários sintomas, o agente os coloca em uma fila e investiga cada um individualmente.
- **Detecção de Emergência**: Identifica palavras-chave de risco (ex: "dor no peito", "falta de ar") e instrui o usuário a procurar ajuda imediata.
- **Geração de Resumo Médico**: Ao final da triagem, cria um resumo profissional para ser anexado ao prontuário do paciente.
- **Persistência de Conversa**: Utiliza o MongoDB para salvar o estado das conversas, permitindo que elas sejam retomadas.

## Arquitetura do Projeto

O fluxo de dados e operações do ClinicAI funciona da seguinte maneira:

1.  **WhatsApp (Evolution API)**: O usuário envia uma mensagem. A Evolution API a captura e envia para o webhook.
2.  **Servidor Flask (`main.py`)**: Recebe o webhook, extrai a mensagem e o número do remetente.
3.  **Agente LangGraph (`agent.py`)**: O servidor invoca o agente com a nova mensagem. O agente:
    - Recupera o estado da conversa atual do MongoDB.
    - Utiliza um roteador (`triage_router`) para decidir qual nó executar a seguir (coletar dados gerais, detalhar um sintoma, gerar resumo ou acionar o protocolo de emergência).
    - Os nós (`nodes.py`) usam o LLM Gemini com prompts específicos (`prompts.py`) para formular respostas e extrair dados.
    - Atualiza o estado da conversa com as novas informações.
4.  **MongoDB**: O estado da conversa é salvo a cada passo. O resumo final também é armazenado em uma coleção separada.
5.  **Resposta ao Usuário**: A resposta gerada pelo agente é enviada de volta ao usuário via API do WhatsApp.

## Tecnologias Utilizadas

- **Backend**: Flask
- **IA e Orquestração**: LangChain, LangGraph
- **Modelo de Linguagem**: Google Gemini 2.5 Flash
- **Banco de Dados**: MongoDB (com PyMongo e MongoDBSaver do LangGraph)
- **Comunicação com WhatsApp**: Evolution API
- **Gerenciamento de Ambiente**: `python-dotenv`

## Configuração e Instalação

**Pré-requisitos:**
- Python 3.10.12
- Ngrok
- Acesso a um servidor MongoDB
- Uma instância da Evolution API configurada

1.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` na raiz do projeto, baseado no arquivo `main.py`:
    ```env
    EVOLUTION_API_URL="http://localhost:8080"
    EVOLUTION_API_KEY="sua-api-key"
    EVOLUTION_INSTANCE_NAME="default"
    
    MONGO_URI="mongodb+srv://..."
    
    GOOGLE_API_KEY="sua-google-api-key"
    ```

3.  **Execute o servidor Flask:**
    ```bash
    python main.py
    ```
    O servidor estará rodando em `http://localhost:5001`. Use uma ferramenta como o `ngrok` para expor esta porta à internet e configurar o webhook na sua Evolution API.

## Estrutura de Arquivos

```
.
├── app/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py         # Definição do grafo e do estado do agente
│   │   ├── nodes.py         # Lógica dos nós do grafo (triage, extract, summarize)
│   │   └── prompts.py       # Todos os prompts usados pelo LLM
│   └── __init__.py
├── main.py                  # Ponto de entrada da aplicação (servidor Flask e webhook)
├── .env                     # Arquivo para variáveis de ambiente (não versionado)
└── requirements.txt         # Lista de dependências Python
```
