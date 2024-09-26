import chromadb
from chromadb.config import Settings
import pandas as pd
import PyPDF2
import os
import hashlib
import json
from sentence_transformers import SentenceTransformer
import torch
import re
import numpy as np
import chroma_setup
from chroma_setup import NOME_COLECAO, client, collection


""" # Nome da coleção definido como uma constante global
NOME_COLECAO = "my_collection" """


# Acesso ao cliente e à coleção
client = chroma_setup.client
collection = chroma_setup.collection

# Verifica se o cliente já existe, caso contrário, cria um novo
try:
    client
except NameError:
    client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))

# Verifica se a coleção já existe, caso contrário, cria uma nova
try:
    collection = client.get_collection(name=NOME_COLECAO)
except ValueError:
    collection = client.create_collection(
        name=NOME_COLECAO,
        metadata={"hnsw:space": "cosine"}  # Configura para usar similaridade do cosseno
    )

# Inicializa o modelo de embeddings
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

MAX_TOKENS = 480  # máximo de 512
OVERLAP = 160  #sobreposição

def preprocessar_texto(texto):
    # Converte para minúsculas
    texto = texto.lower()
    # Remove caracteres especiais, mantendo pontuação básica
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s.,!?]', '', texto)
    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def chunkar_texto(texto, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), max_tokens - overlap):
        chunk = ' '.join(palavras[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def gerar_embeddings(texto):
    texto_preprocessado = preprocessar_texto(texto)
    chunks = chunkar_texto(texto_preprocessado)
    embeddings = model.encode(chunks)
    
    # Implementa uma estratégia de ponderação mais sofisticada
    pesos = np.linspace(1.0, 1.5, len(chunks))  # Peso crescente para chunks posteriores
    pesos[0] *= 1.2  # Aumenta o peso do primeiro chunk
    pesos[-1] *= 1.2  # Aumenta o peso do último chunk
    
    embeddings_ponderados = embeddings * pesos[:, np.newaxis]
    return np.mean(embeddings_ponderados, axis=0).tolist()

def ler_csv(arquivo):
    # Lê o CSV sem usar a primeira linha como cabeçalho
    df = pd.read_csv(arquivo, header=None, keep_default_na=False)
    
    # Cria nomes de colunas genéricos
    df.columns = [f'Coluna_{i}' for i in range(len(df.columns))]
    
    # Remove colunas completamente vazias
    df = df.dropna(axis=1, how='all')
    
    # Imprime informações sobre o DataFrame
    print(f"\nInformações sobre o arquivo {arquivo}:")
    print(f"Número de linhas: {len(df)}")
    print(f"Número de colunas: {len(df.columns)}")
    
    # Converte para dicionário
    records = df.to_dict(orient='records')
    
    # Imprime o primeiro registro para inspeção
    print("\nPrimeiro registro:")
    for key, value in records[0].items():
        if value:  # Imprime apenas valores não vazios
            print(f"{key}: {value}")
    
    return records

def ler_pdf(arquivo):
    with open(arquivo, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texto_completo = ""
        for pagina in reader.pages:
            texto_completo += pagina.extract_text() + " "
    
    # Pré-processamento do texto
    texto_preprocessado = preprocessar_texto(texto_completo)
    
    # Divide o texto em chunks
    chunks = chunkar_texto(texto_preprocessado)
    
    # Cria um documento para cada chunk
    documentos = [{"conteudo": chunk, "fonte": f"{arquivo}_chunk_{i}"} for i, chunk in enumerate(chunks)]
    
    return documentos

def calcular_hash(conteudo):
    return hashlib.md5(json.dumps(conteudo, sort_keys=True).encode()).hexdigest()

def inserir_documentos(documentos, nome_arquivo):
    documentos_inseridos = 0
    documentos_existentes = 0
    for doc in documentos:
        # Remove pares chave-valor com valores vazios
        doc = {k: v for k, v in doc.items() if v != ''}
        texto_completo = ' '.join(str(v) for v in doc.values() if v is not None)
        hash_doc = calcular_hash(doc)
        id_doc = f"{nome_arquivo}_{hash_doc}"
        
        # Verifica se o documento já existe no banco de dados
        resultados = collection.get(ids=[id_doc])
        if not resultados['ids']:
            embeddings = gerar_embeddings(texto_completo)
            collection.add(
                documents=[json.dumps(doc)],
                embeddings=[embeddings],
                metadatas=[{"source": nome_arquivo}],
                ids=[id_doc]
            )
            documentos_inseridos += 1
        else:
            documentos_existentes += 1
    
    return documentos_inseridos, documentos_existentes

# Lê e insere documentos CSV e PDF do diretório 'docs'
diretorio_docs = 'docs'
total_inseridos = 0
total_existentes = 0

for arquivo in os.listdir(diretorio_docs):
    caminho_arquivo = os.path.join(diretorio_docs, arquivo)
    if arquivo.endswith('.csv'):
        documentos = ler_csv(caminho_arquivo)
        inseridos, existentes = inserir_documentos(documentos, arquivo)
    elif arquivo.endswith('.pdf'):
        documentos = ler_pdf(caminho_arquivo)
        for i, doc in enumerate(documentos):
            inseridos, existentes = inserir_documentos([doc], f"{arquivo}_chunk_{i}")
            total_inseridos += inseridos
            total_existentes += existentes
    else:
        print(f"Arquivo não suportado: {arquivo}")
        continue
    
    total_inseridos += inseridos
    total_existentes += existentes
    print(f"Arquivo {arquivo}: {inseridos} documentos inseridos, {existentes} já existentes.")

print(f"\nResumo do processamento:")
print(f"Total de documentos inseridos: {total_inseridos}")
print(f"Total de documentos já existentes: {total_existentes}")

if total_inseridos == 0 and total_existentes > 0:
    print("Parece que todos os documentos já foram inseridos anteriormente.")
elif total_inseridos == 0 and total_existentes == 0:
    print("Nenhum documento foi processado. Verifique se há arquivos válidos na pasta 'docs'.")