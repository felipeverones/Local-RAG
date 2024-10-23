from chromadb.config import Settings
import json
import os
from sentence_transformers import SentenceTransformer
import torch
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chroma_setup import NOME_COLECAO, client, collection
import chroma_setup
import config
from carregar_modelo import carregar_modelo

# Cria um cliente persistente e obtém a coleção
client = chroma_setup.client
collection = chroma_setup.collection

#Inicializa o modelo de embeddings
model, tokenizer = carregar_modelo(config.MODELO_EMBEDDINGS)

MAX_TOKENS = int(config.MAX_TOKENS)  # máximo de 512
OVERLAP = int(config.OVERLAP)  #sobreposição

def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s.,!?0-9]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extrair_numeros(texto):
    return re.findall(r'\d+(?:\.\d+)?', texto)

def calcular_pontuacao_personalizada(query, doc):
    query_preprocessada = preprocessar_texto(query)
    doc_preprocessado = preprocessar_texto(json.dumps(doc))

    # Calcula a similaridade de cosseno
    query_embedding = gerar_embeddings(query_preprocessada)
    doc_embedding = gerar_embeddings(doc_preprocessado)
    similaridade_cosseno = cosine_similarity([query_embedding], [doc_embedding])[0][0]

    # Verifica correspondências exatas de números
    query_numeros = set(extrair_numeros(query))
    doc_numeros = set(extrair_numeros(doc_preprocessado))
    correspondencias_numeros = len(query_numeros.intersection(doc_numeros))

    # Verifica correspondências exatas de palavras-chave
    palavras_chave_query = set(query_preprocessada.split())
    palavras_chave_doc = set(doc_preprocessado.split())
    correspondencias_palavras = len(palavras_chave_query.intersection(palavras_chave_doc))

    # Calcula a pontuação final
    pontuacao = (
        similaridade_cosseno * 0.5 +
        correspondencias_numeros * 0.3 +
        correspondencias_palavras * 0.2
    )

    return pontuacao

def chunkar_texto(texto, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), max_tokens - overlap):
        chunk = ' '.join(palavras[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def gerar_embeddings(texto):
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
                chunk_embeddings = torch.mean(chunk_embeddings, dim=1).numpy()
                embeddings.append(chunk_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)

        return np.mean(embeddings, axis=0).tolist()
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return None

def expandir_consulta(texto):
    # Simples expansão de consulta
    palavras = texto.split()
    expansao = []
    for palavra in palavras:
        if len(palavra) > 4:
            expansao.append(palavra[:-1])  # Adiciona a palavra sem a última letra
    return ' '.join(palavras + expansao)

def pesquisar_descricao(texto_pesquisa):
    texto_expandido = expandir_consulta(texto_pesquisa)
    query_embeddings = gerar_embeddings(texto_expandido)
    resultados = collection.query(
        query_embeddings=[query_embeddings],
        n_results=5  # Aumentamos o número de resultados para refinar depois
    )
    if resultados['distances'][0]:
        documentos = []
        for doc_str, distance in zip(resultados['documents'][0], resultados['distances'][0]):
            try:
                doc = json.loads(doc_str)
            except json.JSONDecodeError:
                doc = {"conteudo": doc_str}

            # Calculamos a pontuação personalizada
            pontuacao = calcular_pontuacao_personalizada(texto_pesquisa, doc)

            doc["score"] = pontuacao
            documentos.append(doc)
        return sorted(documentos, key=lambda x: x['score'], reverse=True)[:5]  # Retornamos os 5 melhores resultados
    else:
        return None

def exibir_resultado(resultado):
    similaridade = resultado['score']
    print(f"\nSimilaridade: {similaridade:.2%}")
    for chave, valor in resultado.items():
        if chave != 'score':
            print(f"{chave}: {valor}")

# Interface de usuário simples
while True:
    texto_pesquisa = input("Digite o texto para pesquisar (ou 'sair' para encerrar): ")

    if texto_pesquisa.lower() == 'sair':
        break

    resultados = pesquisar_descricao(texto_pesquisa)

    if resultados:
        print("\nResultados encontrados:")
        for i, resultado in enumerate(resultados, 1):
            print(f"\nResultado {i}:")
            exibir_resultado(resultado)
    else:
        print("Nenhum resultado encontrado.")

    print("\n" + "-"*50 + "\n")

print("Programa encerrado.")
