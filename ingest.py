from chromadb.config import Settings
import pandas as pd
import PyPDF2
import os
import hashlib
import json
import torch
import re
import numpy as np
import chroma_setup
from chroma_setup import NOME_COLECAO, client, collection
import config
from carregar_modelo import carregar_modelo
from extrair_tabela import extrair_tabela_pdf

# Acesso ao cliente e à coleção
client = chroma_setup.client
collection = chroma_setup.collection

# Inicializa o modelo de embeddings e variáveis de ambiente
model, tokenizer = carregar_modelo(config.MODELO_EMBEDDINGS)

MAX_TOKENS = int(config.MAX_TOKENS)  # máximo de 512
OVERLAP = int(config.OVERLAP)  # sobreposição


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
    i = 0
    while i < len(palavras):
        chunk = palavras[i:i + max_tokens]
        chunks.append(' '.join(chunk))
        i += max_tokens - overlap
    return chunks

def gerar_embeddings_para_documento(documento):
    if isinstance(documento, pd.DataFrame):
        return gerar_embeddings_para_dataframe(documento)
    elif isinstance(documento, str):
        return gerar_embeddings_para_texto(documento)
    elif isinstance(documento, dict):
        return gerar_embeddings_para_dict(documento)
    else:
        print(f"Tipo de documento não suportado: {type(documento)}")
        return None

def gerar_embeddings_para_dataframe(df):
    embeddings = []
    for coluna in df.columns:
        texto_coluna = ' '.join(df[coluna].astype(str).fillna('').tolist())
        if not texto_coluna or len(texto_coluna.strip()) == 0:
            print(f"Aviso: Texto vazio encontrado na coluna {coluna}, pulando geração de embedding")
            continue

        texto_preprocessado = preprocessar_texto(texto_coluna)
        chunks = chunkar_texto(texto_preprocessado)

        if not chunks:
            print(f"Aviso: Nenhum chunk gerado na coluna {coluna}, pulando geração de embedding")
            continue

        try:
            if tokenizer is None:
                embeddings_coluna = model.encode(chunks)
            else:
                # Usar AutoModel para gerar embeddings
                embeddings_coluna = []
                for chunk in chunks:
                    tokens = tokenizer(chunk, return_tensors="pt").to(config.DEVICE)
                    with torch.no_grad():
                        chunk_embeddings = model(**tokens).last_hidden_state
                    chunk_embeddings = torch.mean(chunk_embeddings, dim=1).cpu().numpy()  # Movendo para CPU antes de converter para NumPy
                    embeddings_coluna.append(chunk_embeddings)
                embeddings_coluna = np.concatenate(embeddings_coluna, axis=0)

            embeddings.append(np.mean(embeddings_coluna, axis=0).tolist())
        except Exception as e:
                print(f"Erro ao gerar embeddings: {e}")
                return None
    return embeddings

def gerar_embeddings_para_texto(texto):
    if not texto or len(texto.strip()) == 0:
        print(f"Aviso: Texto vazio encontrado, pulando geração de embedding")
        return None

    texto_preprocessado = preprocessar_texto(texto)
    chunks = chunkar_texto(texto_preprocessado)

    if not chunks:
        print(f"Aviso: Nenhum chunk gerado após preprocessamento")
        return None

    try:
        if tokenizer is None:
            embeddings = model.encode(chunks)
        else:
            # Usar AutoModel para gerar embeddings
            embeddings = []
            for chunk in chunks:
                tokens = tokenizer(chunk, return_tensors="pt").to(config.DEVICE)
                with torch.no_grad():
                    chunk_embeddings = model(**tokens).last_hidden_state
                chunk_embeddings = torch.mean(chunk_embeddings, dim=1).cpu().numpy()  # Movendo para CPU antes de converter para NumPy
                embeddings.append(chunk_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)

        return np.mean(embeddings, axis=0).tolist()
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return None

def gerar_embeddings_para_dict(documento):
    texto_completo = ' '.join(str(v) for v in documento.values() if v is not None)
    if not texto_completo.strip():
        print(f"Aviso: Texto vazio encontrado, pulando geração de embedding")
        return None

    texto_preprocessado = preprocessar_texto(texto_completo)
    chunks = chunkar_texto(texto_preprocessado)

    if not chunks:
        print(f"Aviso: Nenhum chunk gerado após preprocessamento")
        return None

    try:
        if tokenizer is None:
            embeddings = model.encode(chunks)
        else:
            # Usar AutoModel para gerar embeddings
            embeddings = []
            for chunk in chunks:
                tokens = tokenizer(chunk, return_tensors="pt").to(config.DEVICE)
                ##print(f"Tokens: {tokens}")  # Adiciona um print para exibir os tokens
                with torch.no_grad():
                    chunk_embeddings = model(**tokens).last_hidden_state
                chunk_embeddings = torch.mean(chunk_embeddings, dim=1).cpu().numpy()  # Movendo para CPU antes de converter para NumPy
                embeddings.append(chunk_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)

        return np.mean(embeddings, axis=0).tolist()
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return None

""" def ler_csv(arquivo):
    # Lê o CSV usando a primeira linha como cabeçalho
    df = pd.read_csv(arquivo,
                     header=0,
                     keep_default_na=False,
                     on_bad_lines='skip')

    # Preenche células vazias com uma string vazia para garantir que todas as linhas tenham o mesmo número de colunas
    df = df.apply(lambda row: row.fillna(''), axis=1)

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

    return records """
    
def ler_csv(arquivo):
    try:
        # Handle both file path and file object cases
        if isinstance(arquivo, str):
            df = pd.read_csv(arquivo, header=0, keep_default_na=False, on_bad_lines='skip')
        else:
            # For uploaded files through Streamlit
            df = pd.read_csv(arquivo, header=0, keep_default_na=False, on_bad_lines='skip')
        
        # Fill NA values and drop empty columns
        df = df.fillna('')
        df = df.dropna(axis=1, how='all')
        print(df.head())  # Print the first few rows of the DataFrame for debugging
        
        # Convert DataFrame to list of documents
        documentos = []
        for _, row in df.iterrows():
            # Create a document for each row
            documento = {
                "conteudo": " ".join(str(val) for val in row if val),  # Combine all non-empty values
                "metadados": {col: str(val) for col, val in row.items() if val}  # Store original values as metadata
            }
            documentos.append(documento)
        
        print(f"\nProcessamento CSV:")
        print(f"Número de linhas processadas: {len(documentos)}")
        print("Exemplo de documento:", documentos[0] if documentos else "Nenhum documento gerado")
        
        return documentos
        
    except Exception as e:
        print(f"Erro ao processar CSV: {e}")
        raise

def ler_pdf(arquivo):
    if isinstance(arquivo, str):
        with open(arquivo, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            texto_completo = ""
            for pagina in reader.pages:
                texto_completo += pagina.extract_text() + " "
    elif hasattr(arquivo, 'read'):
        reader = PyPDF2.PdfReader(arquivo)
        texto_completo = ""
        for pagina in reader.pages:
            texto_completo += pagina.extract_text() + " "
    else:
        raise TypeError("expected str, bytes or os.PathLike object, not UploadedFile")

    # Extrai tabelas do PDF
    tabelas = extrair_tabela_pdf(arquivo)

    # Converte tabelas para texto
    texto_tabelas = ""
    for tabela in tabelas:
        texto_tabelas += tabela.to_string(index=False) + " "

    # Combina texto do PDF com texto das tabelas
    texto_completo += texto_tabelas

    # Pré-processamento do texto
    texto_preprocessado = preprocessar_texto(texto_completo)

    # Divide o texto em chunks
    chunks = chunkar_texto(texto_preprocessado)

    # Cria um documento para cada chunk
    documentos = [{"conteudo": chunk, "fonte": f"{arquivo}_chunk_{i}"} for i, chunk in enumerate(chunks)]

    # Imprime informações sobre o PDF
    print(f"\nInformações sobre o arquivo {arquivo}:")
    print(f"Número de páginas: {len(reader.pages)}")
    print(f"Número de chunks gerados: {len(chunks)}")

    return documentos

def calcular_hash(conteudo):
    return hashlib.md5(json.dumps(conteudo, sort_keys=True).encode()).hexdigest()

def inserir_documentos(documentos, nome_arquivo):
    """
    Insere documentos no ChromaDB com melhor tratamento de erros e logging
    """
    from datetime import datetime
    import uuid
    
    # Acesso ao cliente e à coleção
    client = chroma_setup.client
    collection = chroma_setup.collection
    
    documentos_inseridos = 0
    documentos_existentes = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    
    for idx, doc in enumerate(documentos):
        try:
            # Garantir que temos conteúdo para processar
            if isinstance(doc, dict) and "conteudo" in doc:
                texto_completo = doc["conteudo"]
            else:
                texto_completo = str(doc)  # Fallback para caso o documento não seja um dicionário

            if not texto_completo.strip():
                print(f"Aviso: Documento {idx} vazio, pulando...")
                continue

            id_doc = f"{nome_arquivo}_{timestamp}_{session_id}_{idx}"
            
            # Gerar embeddings
            embeddings = gerar_embeddings_para_documento(doc)
            
            if embeddings is not None:
                # Preparar metadados
                metadata = {
                    "source": nome_arquivo,
                    "timestamp": timestamp,
                    "chunk_index": idx
                }
                
                # Se o documento tem metadados próprios, incluí-los
                if isinstance(doc, dict) and "metadados" in doc:
                    metadata.update(doc["metadados"])
                
                # Inserir no ChromaDB
                collection.add(
                    documents=[json.dumps(doc)],
                    embeddings=[embeddings],
                    metadatas=[metadata],
                    ids=[id_doc]
                )
                
                documentos_inseridos += 1
                print(f"Documento {id_doc} inserido com sucesso")
            else:
                print(f"Aviso: Não foi possível gerar embeddings para o documento {id_doc}")
                documentos_existentes += 1
                
        except Exception as e:
            print(f"Erro ao processar documento {idx}: {str(e)}")
            documentos_existentes += 1
            continue
            
    return documentos_inseridos, documentos_existentes


def limpar_documentos_fonte(nome_arquivo):
    """
    Remove todos os documentos anteriores com a mesma fonte antes de inserir novos.
    
    Args:
        nome_arquivo (str): Nome do arquivo/fonte a ser limpo
    """
    try:
        collection.delete(
            where={"source": nome_arquivo}
        )
        print(f"Documentos anteriores de {nome_arquivo} removidos com sucesso.")
    except Exception as e:
        print(f"Erro ao tentar limpar documentos anteriores: {e}")


#####################################################################################
# Funções de processamento de texto e embeddings

def ingest_documents():
    # Lê e insere documentos CSV e PDF do diretório
    diretorio_docs = config.DIRETORIO_DOCS
    total_inseridos = 0
    total_existentes = 0
    
    # Verifica se o diretório existe, e cria se não existir
    if not os.path.exists(diretorio_docs):
        os.mkdir(diretorio_docs)
        print(f"Diretório '{diretorio_docs}' criado.")

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
    return 0

if __name__ == "__main__":
    ingest_documents()
    

