import chromadb
from chromadb.config import Settings
import json

# Cria um cliente persistente
client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))

# Nome da coleção
NOME_COLECAO = "my_collection"

# Obtém a coleção
collection = client.get_collection(name=NOME_COLECAO)

def pesquisar_descricao(texto_pesquisa):
    # Realiza a pesquisa na coleção
    resultados = collection.query(
        query_texts=[texto_pesquisa],
        n_results=5  # Retorna os 5 resultados mais similares
    )

    if resultados['distances'][0]:
        documentos = []
        for doc_str, score in zip(resultados['documents'][0], resultados['distances'][0]):
            try:
                doc = json.loads(doc_str)
            except json.JSONDecodeError:
                doc = {"conteudo": doc_str}
            doc["score"] = score
            documentos.append(doc)
        return documentos
    else:
        return None

def exibir_resultado(resultado):
    print(f"\nScore: {resultado['score']:.4f}")
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