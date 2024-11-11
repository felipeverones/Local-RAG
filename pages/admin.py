import streamlit as st
import os
from ingest import inserir_documentos, ler_csv, ler_pdf, ingest_documents  # Certifique-se de que essas funções estão disponíveis
from delete_db import deletar_banco_dados

# Configuração da página deve ser a primeira chamada do Streamlit
st.set_page_config(
    page_title="Administração RAG",
    page_icon="⚙️",
    initial_sidebar_state="expanded")

st.title("Administração do Sistema RAG")


# Função para inserir documentos
def inserir_documentos_interface():
    st.header("Inserir Documentos")
    uploaded_files = st.file_uploader("Escolha arquivos CSV ou PDF", type=["csv", "pdf"], accept_multiple_files=True)

    if st.button("Inserir Documentos"):
        if uploaded_files:
            total_inseridos = 0
            total_existentes = 0
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.csv'):
                    # Lógica para ler e inserir CSV
                    documentos = ler_csv(uploaded_file)
                    inseridos, existentes = inserir_documentos(documentos, uploaded_file.name)
                elif uploaded_file.name.endswith('.pdf'):
                    # Lógica para ler e inserir PDF
                    documentos = ler_pdf(uploaded_file)
                    for i, doc in enumerate(documentos):
                        inseridos, existentes = inserir_documentos([doc], f"{uploaded_file.name}_chunk_{i}")
                total_inseridos += inseridos
                total_existentes += existentes
            
            st.success(f"Total de documentos inseridos: {total_inseridos}, já existentes: {total_existentes}")
        else:
            st.warning("Por favor, carregue pelo menos um arquivo.")


# Chamar as funções de interface
inserir_documentos_interface()


# Botão para deletar banco de dados
if st.button("Deletar Banco de Dados"):
    with st.spinner("Deletando banco de dados..."):
        deletar_banco_dados()
    st.success("Banco de dados deletado com sucesso.")

if st.button("Atualizar Banco de dados"):
    with st.spinner("Atualizando banco de dados..."):
        ingest_documents()
    st.success("Banco de dados atualizado com sucesso!")