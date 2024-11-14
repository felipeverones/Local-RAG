# Projeto Local RAG (Retrieval-Augmented Generation)

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) usando ChromaDB para armazenamento de embeddings e um modelo de linguagem para geração de respostas.

## Funcionalidades

 - Ingestão de documentos nos formatos CSV e PDF para extração de informações e criação de uma base de conhecimento local, privada e segura.
 - Chat com documento, permitindo que o usuário faça perguntas sobre o conteúdo dos documentos inseridos e obtenha respostas contextuais e relevantes em uma interface de chat ágil e intuitiva.
 - Geração de respostas usando um modelo de linguagem integrado ao RAG para fornecer respostas precisas e contextuais com base nas informações armazenadas.
 - Suporte para execução local do modelo de linguagem usando Ollama para otimizar o desempenho e reduzir a latência.
 - Utilização de CUDA Toolkit 12.4 para acelerar a execução do modelo de linguagem e melhorar o desempenho geral do sistema.
 - Administração do banco de dados ChromaDB, permitindo adicionar, remover e atualizar documentos conforme necessário.

## Requisitos

- Python 3.11+
- pip (gerenciador de pacotes do Python)
- Ollama (para execução local dos modelos de linguagem)
- CUDA Toolkit 12.4

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

- `app.py`: Interface Streamlit principal para interação com o usuário.
- `rag_main.py`: Implementação do RAG com integração LLM.
- `ingest.py`: Script para ingestão de documentos CSV e PDF.
- `delete_db.py`: Utilitário para limpar o banco de dados.
- `config.py`: Configurações do projeto, como caminhos de arquivos e parâmetros do modelo LLM.
- `chroma_setup.py`: Configuração do ChromaDB para inicialização do cliente persistente e armazenamento de embeddings e metadados dos documentos.

## Uso

### Modo Desenvolvedor

1. Ingestão de Documentos (3 formas):

   1.1 Coloque seus arquivos CSV e PDF na pasta `docs/` e execute:
   ```
   python ingest.py
   ```
   Ou, se preferir, utilize a interface Streamlit (Passo 2) para fazer o upload dos arquivos, navegando até a página de `admin` e clicando no botão `Atualizar banco de dados`.

   1.2 Utilize a interface Streamlit (Passo 2) para fazer o upload de cada documento na página de `admin`, clicando no botão `Inserir Documentos` e selecionando o arquivo desejado.

   1.3 Utilize a interface Streamlit (Passo 2) para fazer upload do documento através da página de `chat` inicial, clicando no botão `Carregar Documento` e selecionando o arquivo desejado.
   

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

### Modo Usuário Final (Streamlit)

- No Windows, execute o arquivo `Iniciar.bat`, este irá executar os comandos descritos acima

- No Linux, abra o terminal no diretório do projeto e conceda permissões de execução do script com:

   ```
   chmod +x Iniciar.sh
   ```
   Depois, execute o script com o comando abaixo:

   ```
   ./Iniciar.sh
   ```


## Notas Adicionais

- O sistema usa, por padrão, o modelo de processamento de linguagem natural (NLP) `all-mpnet-base-v2` para gerar embeddings.
- O LLM utilizado é o Llama 3.1, executado localmente via Ollama.
- Os documentos são armazenados e indexados usando ChromaDB.
- O modelo deve ser instalado localmente.
