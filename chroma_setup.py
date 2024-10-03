import os
import chromadb
from chromadb.config import Settings  # Certifique-se de que essas funções estão disponíveis
import config



NOME_COLECAO = config.NOME_COLECAO  # Nome da coleção
DIRETORIO_DOCS = config.DIRETORIO_DOCS  # Diretório onde os documentos estão armazenados

def inicializar_chroma():
    """Inicializa o cliente ChromaDB persistente com configuração para permitir reset e cria a coleção se não existir."""
    client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))

    try:
        collection = client.get_collection(name=NOME_COLECAO)
        print(f"Coleção encontrada! {NOME_COLECAO}")
    except ValueError:
        collection = client.create_collection(
            name=NOME_COLECAO,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Coleção {NOME_COLECAO} criada!")
    
    return client, collection


# Inicializa o ChromaDB ao importar o módulo
client, collection = inicializar_chroma()

