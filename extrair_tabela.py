import pandas as pd
import pdfplumber
from tabulate import tabulate

def encontrar_inicio_tabela(linhas):
    """
    Encontra o índice da linha que provavelmente inicia uma tabela no texto extraído do PDF.
    """
    for i, linha in enumerate(linhas):
        # Verifica se a linha contém um número significativo de colunas separadas por espaços
        # e se a linha não está vazia ou contém apenas um único elemento.
        if len(linha.split()) > 2 and linha.strip():
            return i
    return None

def extrair_tabela_pdf(caminho_arquivo):
    """
    Extrai todas as tabelas de um arquivo PDF.
    """
    tabelas = []
    with pdfplumber.open(caminho_arquivo) as pdf:
        for page in pdf.pages:
            texto = page.extract_text()
            linhas = texto.split('\n')

            inicio_tabela = encontrar_inicio_tabela(linhas)

            if inicio_tabela is not None:
                tabela_linhas = linhas[inicio_tabela:]
                tabela_dados = []
                for linha in tabela_linhas:
                    # Remove espaços em branco extras e divide a linha em colunas
                    colunas = linha.strip().split()
                    if colunas:
                        tabela_dados.append(colunas)

                df = pd.DataFrame(tabela_dados)
                tabelas.append(df)

    return tabelas

# Caminho do arquivo PDF
caminho_arquivo = 'docs/Licitações 2024.pdf'

# Extrair e exibir a tabela
extrair_tabela_pdf(caminho_arquivo)
