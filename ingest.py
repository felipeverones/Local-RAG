import chromadb
from chromadb.config import Settings
import pandas as pd
import PyPDF2
import os
import hashlib
import json

# Nome da coleção definido como uma constante global
NOME_COLECAO = "my_collection"

# Verifica se o cliente já existe, caso contrário, cria um novo
try:
    client
except NameError:
    client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))

# Verifica se a coleção já existe, caso contrário, cria uma nova
try:
    collection = client.get_collection(name=NOME_COLECAO)
except ValueError:
    collection = client.create_collection(name=NOME_COLECAO)

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
        texto = ""
        for pagina in reader.pages:
            texto += pagina.extract_text()
    return [{"conteudo": texto}]

def calcular_hash(conteudo):
    return hashlib.md5(json.dumps(conteudo, sort_keys=True).encode()).hexdigest()

def inserir_documentos(documentos, nome_arquivo):
    documentos_inseridos = 0
    documentos_existentes = 0
    for doc in documentos:
        # Remove pares chave-valor com valores vazios
        doc = {k: v for k, v in doc.items() if v != ''}
        hash_doc = calcular_hash(doc)
        id_doc = f"{nome_arquivo}_{hash_doc}"
        
        # Verifica se o documento já existe no banco de dados
        resultados = collection.get(ids=[id_doc])
        if not resultados['ids']:
            collection.add(
                documents=[json.dumps(doc)],  # Converte para JSON string
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
        inseridos, existentes = inserir_documentos(documentos, arquivo)
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

#client.heartbeat() # retorna um heartbeat em nanossegundos. Útil para garantir que o cliente permaneça conectado.

#client.reset() # Esvazia e redefine completamente o banco de dados. ⚠️ Isso é destrutivo e não reversível.