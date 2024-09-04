import chromadb
from chromadb.config import Settings
import json
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Suprimir avisos
import warnings
warnings.filterwarnings("ignore")

# Configurações
client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))
NOME_COLECAO = "my_collection"
collection = client.get_collection(name=NOME_COLECAO)
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
MAX_TOKENS = 480
OVERLAP = 160
nome_modelo = "llama3.1:latest"

def configurar_llm():
    print(f"Iniciando o modelo {nome_modelo}...")
    llm = Ollama(model=nome_modelo)
    print(f"Modelo {nome_modelo} iniciado com sucesso!")
    return llm

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
        return "Desculpe, não encontrei informações relevantes para sua pergunta.", None

def main():
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
    main()