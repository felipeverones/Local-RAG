import requests
import os 
from chromadb.config import Settings
import json
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import chroma_setup
from chroma_setup import NOME_COLECAO
import config


# Suprimir avisos
import warnings
warnings.filterwarnings("ignore")


# Acesso ao cliente e à coleção
client = chroma_setup.client
collection = chroma_setup.collection


# Acesso às variáveis de configuração
model = SentenceTransformer(config.MODELO_EMBEDDINGS)
MAX_TOKENS = int(config.MAX_TOKENS)
OVERLAP = int(config.OVERLAP)
nome_modelo = config.MODELO_OLLAMA
OLLAMA_API_URL = config.OLLAMA_API_URL


def verificar_conexao_ollama():
    """Verifica se o Ollama está rodando e responde na URL da API."""
    try:
        resposta = requests.post(OLLAMA_API_URL, json={"model": nome_modelo, "prompt": "Olá"})
        resposta.raise_for_status() #isso lança uma exceção se a resposta não for 200 (ok)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erro ao verificar conexão com o Ollama: {e}")
        return False

def configurar_llm():
    """Configura o LLM Ollama se a conexão for bem-sucedida."""
    if verificar_conexao_ollama():
        print(f"Iniciando o modelo {nome_modelo}...")
        llm = Ollama(model=nome_modelo)
        print(f"Modelo {nome_modelo} iniciado com sucesso!")
        return llm
    else:
        print(f"Não foi possível conectar ao Ollama. Verifique se o Ollama está rodando e se a URL da API está correta.")
        return None

def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s.,!?0-9]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def extrair_numeros(texto):
    return re.findall(r'\d+(?:\.\d+)?', texto)

def calcular_pontuacao_personalizada(query, doc):
    query_preprocessada = preprocessar_texto(query)
    doc_preprocessado = preprocessar_texto(json.dumps(doc))
    
    query_embedding = model.encode([query_preprocessada])[0]
    doc_embedding = model.encode([doc_preprocessado])[0]
    similaridade_cosseno = cosine_similarity([query_embedding], [doc_embedding])[0][0]
    
    query_numeros = set(extrair_numeros(query))
    doc_numeros = set(extrair_numeros(doc_preprocessado))
    correspondencias_numeros = len(query_numeros.intersection(doc_numeros))
    
    palavras_chave_query = set(query_preprocessada.split())
    palavras_chave_doc = set(doc_preprocessado.split())
    correspondencias_palavras = len(palavras_chave_query.intersection(palavras_chave_doc))
    
    pontuacao = (
        similaridade_cosseno * 0.5 +
        correspondencias_numeros * 0.3 +
        correspondencias_palavras * 0.2
    )
    
    return float(pontuacao)

def expandir_consulta(texto):
    palavras = texto.split()
    expansao = []
    for palavra in palavras:
        if len(palavra) > 4:
            expansao.append(palavra[:-1])
    return ' '.join(palavras + expansao)

def pesquisar_descricao(texto_pesquisa):
    texto_expandido = expandir_consulta(texto_pesquisa)
    query_embeddings = model.encode([preprocessar_texto(texto_expandido)])[0].tolist()
    resultados = collection.query(
        query_embeddings=[query_embeddings],
        n_results=5
    )
    if resultados['distances'][0]:
        documentos = []
        for doc_str, distance in zip(resultados['documents'][0], resultados['distances'][0]):
            try:
                doc = json.loads(doc_str)
            except json.JSONDecodeError:
                doc = {"conteudo": doc_str}
            
            pontuacao = calcular_pontuacao_personalizada(texto_pesquisa, doc)
            
            doc["score"] = pontuacao
            documentos.append(doc)
        return sorted(documentos, key=lambda x: x['score'], reverse=True)[:5]
    else:
        return None

prompt_template = """
Você é um assistente de busca amigável e prestativo. Com base nos resultados da busca fornecidos, responda à pergunta do usuário de maneira conversacional e orgânica. Apresente as informações relevantes encontradas, mencionando a similaridade quando apropriado. Se não houver informações suficientes, diga isso de forma educada.

Contexto (resultados da busca):
{context}

Pergunta do usuário: {question}

Resposta:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def gerar_resposta(query, contexto, llm):
    resposta = llm.invoke(prompt.format(context=contexto, question=query))
    return resposta

def pesquisar_e_responder(texto_pesquisa, llm):
    print("Realizando busca...")
    resultados = pesquisar_descricao(texto_pesquisa)
    
    if resultados:
        print("Busca concluída. Gerando resposta...")
        contexto = json.dumps(resultados, ensure_ascii=False, indent=2)
        resposta = gerar_resposta(texto_pesquisa, contexto, llm)
        return resposta, resultados
    else:
        return "Desculpe, não encontrei informações relevantes para sua pergunta. Verifique seu banco de dados e tente novamente.", None

# IMPORTANTE: Descomente para rodar o chat no terminal!!!!!!!!!!!!!!!!!!

""" def main():
    llm = configurar_llm()

    print("Assistente: Olá! Estou aqui para ajudar com suas buscas. O que você gostaria de saber? (Digite 'sair' para encerrar)")

    while True:
        texto_pesquisa = input("\nVocê: ")
        
        if texto_pesquisa.lower() == 'sair':
            print("\nAssistente: Foi um prazer ajudar! Até a próxima.")
            break
        
        resposta, resultados = pesquisar_e_responder(texto_pesquisa, llm)
        
        print(f"\nAssistente: {resposta}")
        
        if not resultados:
            print("Não foram encontrados resultados para esta busca.")
        
        print("\n" + "-"*50)

    print("Programa encerrado.")

if __name__ == "__main__":
    main() """