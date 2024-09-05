# Projeto Local RAG (Retrieval-Augmented Generation)

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) usando ChromaDB para armazenamento de embeddings e um modelo de linguagem para geração de respostas.

## Requisitos

- Python 3.11+
- pip (gerenciador de pacotes do Python)
- Ollama (para execução local dos modelos de linguagem)

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/felipeverones/Local-RAG.git
   cd Local-RAG
   ```

2. Crie um ambiente virtual:
   ```
   python -m venv venv
   ```

3. Ative o ambiente virtual:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

5. Baixe o modelo (Por padrão, o modelo utilizado é o Llama 3.1):
   ```
   ollama pull llama3.1:latest
   ```

## Estrutura do Projeto

- `main.py`: Script principal para busca e pontuação personalizada.
- `rag_main.py`: Implementação do RAG com integração LLM.
- `app.py`: Interface Streamlit para interação com o usuário.
- `ingest.py`: Script para ingestão de documentos CSV e PDF.
- `delete_db.py`: Utilitário para limpar o banco de dados.

## Uso

1. Ingestão de Documentos:
   Coloque seus arquivos CSV e PDF na pasta `docs/` e execute:
   ```
   python ingest.py
   ```

2. Executar a Interface Web:
   ```
   streamlit run app.py
   ```

3. Executar no Terminal (opcional):
   Descomente a função `main()` em `rag_main.py` e execute:
   ```
   python rag_main.py
   ```

4. Limpar o Banco de Dados (se necessário):
   ```
   python delete_db.py
   ```

## Notas Adicionais

- O sistema usa o modelo de embedding `distiluse-base-multilingual-cased-v2` para gerar embeddings.
- O LLM utilizado é o Llama 3.1, executado localmente via Ollama.
- Os documentos são armazenados e indexados usando ChromaDB.
- O modelo deve ser instalado localmente.
