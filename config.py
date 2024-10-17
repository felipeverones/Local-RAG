from torch import cuda
# Configurações do ambiente
NOME_COLECAO="my_collection" 
DIRETORIO_DOCS="docs"
MODELO_EMBEDDINGS="all-mpnet-base-v2"  # Modelo com tamanho de sequência maior
DEVICE = "cuda" if cuda.is_available() else "cpu"
device=""
MAX_TOKENS= 256  # Reduzindo o tamanho máximo de sequência para 256
OVERLAP= 160
MODELO_OLLAMA="llama3.1:latest"
OLLAMA_API_URL="http://localhost:11434/api/generate"

# Lista de modelos a serem testados
MODELOS_TESTE = [
    "distiluse-base-multilingual-cased-v2",
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "nomic-ai/nomic-embed-text-v1"
]
