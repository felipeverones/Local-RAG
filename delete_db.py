import chromadb
import os
import shutil
from chromadb.config import Settings
import chroma_setup
from chroma_setup import NOME_COLECAO

def deletar_banco_dados():
    
    # Acesso ao cliente e à coleção
    client = chroma_setup.client
    #collection = chroma_setup.collection
    
    # Cria um cliente persistente com configuração para permitir reset
    #client = chromadb.PersistentClient(path="db", settings=Settings(allow_reset=True))

    try:
        # Tenta obter a coleção
        collection = client.get_collection(name=NOME_COLECAO)
        print(f"Coleção '{NOME_COLECAO}' encontrada. Iniciando o processo de exclusão...")

        # Deleta a coleção
        client.delete_collection(name=NOME_COLECAO)
        print(f"Coleção '{NOME_COLECAO}' foi excluída com sucesso.")

    except ValueError:
        print(f"A coleção '{NOME_COLECAO}' não existe. Nada para excluir.")

    # Reseta o banco de dados
    try:
        client.reset()
        print("O banco de dados foi resetado.")
    except Exception as e:
        print(f"Erro ao resetar o banco de dados: {str(e)}")

    # Exclui o diretório do banco de dados
    db_path = "db"
    if os.path.exists(db_path):
        try:
            for item in os.listdir(db_path):
                item_path = os.path.join(db_path, item)
                if item != "chroma.sqlite3":
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            print(f"O conteúdo do diretório '{db_path}' foi excluído, exceto 'chroma.sqlite3'.")
        except Exception as e:
            print(f"Erro ao excluir o conteúdo do diretório '{db_path}': {str(e)}")
    else:
        print(f"O diretório '{db_path}' não existe.")

    print("O processo de limpeza foi concluído.")
