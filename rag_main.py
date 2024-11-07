import requests
import os
from chromadb.config import Settings
import json
from sentence_transformers import SentenceTransformer
import re
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import chroma_setup
from chroma_setup import NOME_COLECAO
import config
from carregar_modelo import carregar_modelo

# Suprimir avisos
import warnings
warnings.filterwarnings("ignore")

# Acesso ao cliente e à coleção
client = chroma_setup.client
collection = chroma_setup.collection

# Acesso às variáveis de configuração
model, tokenizer = carregar_modelo(config.MODELO_EMBEDDINGS)

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

    query_embedding = gerar_embeddings(query_preprocessada)
    doc_embedding = gerar_embeddings(doc_preprocessado)
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

def pesquisar_descricao(texto_pesquisa, documento_filtro=None):
    """
    Realiza a pesquisa no ChromaDB, opcionalmente filtrando por um documento específico.
    
    Args:
        texto_pesquisa (str): O texto a ser pesquisado
        documento_filtro (str, optional): Nome do documento para filtrar a busca
    """
    texto_expandido = expandir_consulta(texto_pesquisa)
    query_embeddings = [gerar_embeddings(texto_expandido).cpu().tolist()]
    
    # Define o filtro se um documento específico for fornecido
    where_filter = {"source": documento_filtro} if documento_filtro else None
    
    # Realiza a pesquisa com o filtro
    resultados = collection.query(
        query_embeddings=query_embeddings,
        n_results=5,
        where=where_filter
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

            # Extrair os tópicos principais do documento
            topicos = extrair_topicos_principais(doc["conteudo"])
            doc["tópicos"] = topicos

            documentos.append(doc)
        return sorted(documentos, key=lambda x: x['score'], reverse=True)[:5]
    else:
        return None

def extrair_topicos_principais(texto):
    # Implementar a lógica para extrair os tópicos principais do texto
    # Exemplo simples: extrair as primeiras 3 frases como tópicos
    frases = texto.split(".")[:3]
    return [f.strip() for f in frases]

prompt_template = """
Você é um assistente técnico especializado em análise de dados. Com base nos resultados da busca fornecidos, responda à pergunta ou busca do usuário de forma estruturada seguindo estas diretrizes:

1. Comece com uma breve introdução conversacional.

2. Apresente as informações técnicas encontradas em tópicos claros usando este formato:
   • Tópico Principal 1:
     - Subtópico/Detalhe
     - Valores/Métricas relevantes
   • Tópico Principal 2:
     - Subtópico/Detalhe
     - Valores/Métricas relevantes

3. Se houver dados numéricos ou métricas:
   • Apresente-os em formato estruturado
   • Inclua porcentagens, valores e comparações quando disponíveis
   • Destaque tendências ou padrões importantes

4. Para informações técnicas complexas:
   • Divida em partes menores e mais compreensíveis
   • Explique termos técnicos quando necessário
   • Relacione diferentes aspectos quando relevante

5. Se a informação for insuficiente ou incerta:
   • Indique claramente quais aspectos precisam de mais dados
   • Sugira possíveis fontes adicionais de informação

Tópicos principais identificados nos resultados:
{topics}

Contexto (resultados da busca):
{context}

Pergunta do usuário: {question}

Resposta:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["topics", "context", "question"])

def gerar_resposta(query, contexto, llm):
    # Extrair os tópicos principais dos resultados da busca
    resultados_json = json.loads(contexto)
    
    # Melhorar a extração de tópicos
    topicos = set()
    metricas = set()
    termos_tecnicos = set()
    
    for resultado in resultados_json:
        # Extrair tópicos do conteúdo
        if isinstance(resultado, dict):
            conteudo = resultado.get("conteudo", "")
            if isinstance(conteudo, str):
                # Identificar possíveis métricas (números seguidos de unidades comuns)
                metricas.update(re.findall(r'\d+(?:\.\d+)?(?:%|\s*(?:GB|MB|KB|ms|seconds|minutes|hours|days))', conteudo))
                
                # Identificar termos técnicos (palavras com maiúsculas no meio ou acrônimos)
                termos_tecnicos.update(re.findall(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b|\b[A-Z]{2,}\b', conteudo))
                
                # Adicionar tópicos existentes
                topicos.update(resultado.get("tópicos", []))

    # Criar um resumo estruturado dos tópicos encontrados
    topicos_resumo = {
        "Tópicos Principais": list(topicos),
        "Métricas Identificadas": list(metricas),
        "Termos Técnicos": list(termos_tecnicos)
    }
    
    # Converter para string formatada
    topicos_str = json.dumps(topicos_resumo, ensure_ascii=False, indent=2)

    # Gerar a resposta usando o prompt atualizado
    resposta = llm.invoke(prompt.format(
        topics=topicos_str,
        context=contexto,
        question=query
    ))
    return resposta

def extrair_topicos_principais(texto):
    """
    Extrai tópicos principais do texto de forma mais estruturada.
    """
    # Extrair frases que parecem ser títulos ou tópicos principais
    frases = re.split(r'[.!?]\s+', texto)
    topicos = []
    
    for frase in frases:
        frase = frase.strip()
        # Identificar frases que parecem ser tópicos
        if (len(frase.split()) <= 10 and  # Tópicos tendem a ser curtos
            (frase.istitle() or  # Começa com maiúscula
             re.search(r'\d', frase) or  # Contém números
             re.search(r'[A-Z]{2,}', frase))):  # Contém acrônimos
            topicos.append(frase)
    
    # Limitar a 5 tópicos mais relevantes
    return topicos[:5]

def pesquisar_e_responder(texto_pesquisa, llm, documento_filtro=None):
    """
    Realiza a pesquisa e gera uma resposta, opcionalmente filtrando por documento.
    
    Args:
        texto_pesquisa (str): O texto da pergunta
        llm: O modelo de linguagem a ser usado
        documento_filtro (str, optional): Nome do documento para filtrar a busca
    """
    print(f"Realizando busca no documento: {documento_filtro if documento_filtro else 'todos'}")
    resultados = pesquisar_descricao(texto_pesquisa, documento_filtro)

    if resultados:
        print("Busca concluída. Gerando resposta...")
        contexto = json.dumps(resultados, ensure_ascii=False, indent=2)
        resposta = gerar_resposta(texto_pesquisa, contexto, llm)
        return resposta, resultados
    else:
        return "Desculpe, não encontrei informações relevantes para sua pergunta no documento especificado.", None

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
                chunk_embeddings = torch.mean(chunk_embeddings, dim=1).cpu().numpy()  # Movendo para CPU antes de converter para NumPy
                embeddings.append(chunk_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)

        return torch.tensor(np.mean(embeddings, axis=0))
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return None

def chunkar_texto(texto, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), max_tokens - overlap):
        chunk = ' '.join(palavras[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def carregar_documento(caminho_documento):
    """Carrega o documento do caminho especificado."""
    try:
        with open(caminho_documento, 'r') as file:
            conteudo = file.read()
        return conteudo
    except Exception as e:
        print(f"Erro ao carregar o documento: {e}")
        return None

if __name__ == "__main__":

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

    main()
