import streamlit as st
import chroma_setup
from rag_main import configurar_llm, pesquisar_e_responder, carregar_documento
from ingest import inserir_documentos, ler_pdf, ler_csv, limpar_documentos_fonte

st.set_page_config(page_title="Assistente RAG", page_icon="🤖", layout="wide")

# Inicialização do modelo
@st.cache_resource
def load_model():
    try:
        llm = configurar_llm()
        if llm is None:
            return None, "Erro ao conectar ao Ollama. Verifique se o Ollama está rodando."
        return llm, None
    except Exception as e:
        return None, str("Erro ao carregar o modelo: " + str(e))

# Inicializa estados da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documento_atual" not in st.session_state:
    st.session_state.documento_atual = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "upload_state" not in st.session_state:
    st.session_state.upload_state = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Carrega o modelo e verifica se houve erro
llm, error_message = load_model()

if error_message:
    st.error(error_message)
    st.stop()

st.title("Assistente RAG")

# Callbacks para gerenciar estados
def set_processing_state(state: bool):
    st.session_state.is_processing = state

def process_uploaded_file():
    if st.session_state.upload_state and not st.session_state.is_processing:
        file = st.session_state.upload_state
        if file.name not in st.session_state.processed_files:
            set_processing_state(True)
            return True
    return False

# Exibe o histórico de mensagens
# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "resultados" in message:
            best_result = message["resultados"][0]
            st.write("### Principais informações do melhor resultado")
            
            # Formatação dos metadados
            if "metadados" in best_result:
                st.markdown("#### Metadados:")
                metadados = best_result["metadados"]
                for key, value in metadados.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Tópicos
            if "tópicos" in best_result:
                st.markdown("#### Especificações:")
                for topic in best_result["tópicos"]:
                    st.markdown(f"- {topic}")
            
            # Score
            st.markdown(f"#### Pontuação: {best_result['score']:.2f}")
            
            # Expander para resultados detalhados
            with st.expander("Ver resultados completos da busca"):
                for i, resultado in enumerate(message["resultados"], 1):
                    st.markdown(f"### Resultado {i}")
                    
                    # Conteúdo
                    if "conteudo" in resultado:
                        st.markdown("#### Conteúdo:")
                        st.markdown(resultado["conteudo"])
                    
                    # Metadados dentro do expander
                    if "metadados" in resultado:
                        st.markdown("#### Metadados:")
                        for key, value in resultado["metadados"].items():
                            st.markdown(f"**{key}:** {value}")
                    
                    # Separador entre resultados
                    if i < len(message["resultados"]):
                        st.markdown("---")
            
            
# Status do processamento
if st.session_state.is_processing:
    st.info("⏳ Processando documento...")

# Campo de entrada do usuário
prompt = st.chat_input(
    "O que você gostaria de saber?",
    disabled=st.session_state.is_processing
)

if prompt and not st.session_state.is_processing:
    set_processing_state(True)
    
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Pesquisa considerando o documento atual
            documento_filtro = st.session_state.documento_atual
            with st.spinner("Pesquisando..."):
                resposta, resultados = pesquisar_e_responder(
                    prompt, 
                    llm,
                    documento_filtro
                )

            for chunk in resposta.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            if resultados:
                with st.expander("Ver resultados da busca"):
                    for i, resultado in enumerate(resultados, 1):
                        st.write(f"Resultado {i}:")
                        st.json(resultado)

            # Adiciona a resposta do assistente ao histórico
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "resultados": resultados if resultados else []
            })
        finally:
            set_processing_state(False)
            st.rerun()

# Botão para limpar o histórico
if st.button("Limpar Conversa", disabled=st.session_state.is_processing):
    st.session_state.messages = []
    st.session_state.documento_atual = None
    st.session_state.processed_files = set()
    st.session_state.upload_state = None
    st.rerun()


# Container para o upload e processamento
with st.container():
    # Upload do documento
    uploaded_file = st.file_uploader(
        "Carregar Documento",
        type=["txt", "pdf", "docx", "csv"],
        disabled=st.session_state.is_processing,
        key="document_uploader"
    )
    
    # Atualiza o estado do upload
    if uploaded_file is not None:
        st.session_state.upload_state = uploaded_file
    
    # Processa o arquivo se necessário
    if process_uploaded_file():
        progress_placeholder = st.empty()
        result_placeholder = st.empty()
        
        try:
            with progress_placeholder.container():
                with st.spinner(f"Processando {uploaded_file.name}..."):
                    # Limpa documentos anteriores com mesmo nome
                    limpar_documentos_fonte(uploaded_file.name)
                    
                    # Determina o tipo de arquivo e processa adequadamente
                    if uploaded_file.name.endswith('.pdf'):
                        documentos = ler_pdf(uploaded_file)
                        #st.write(f"PDF processado: {len(documentos)} chunks gerados")
                    elif uploaded_file.name.endswith('.csv'):
                        documentos = ler_csv(uploaded_file)
                        #st.write(f"CSV processado: {len(documentos)} linhas convertidas em documentos")
                    else:
                        # Para arquivos de texto
                        texto = uploaded_file.read().decode('utf-8')
                        documentos = [{"conteudo": texto}]

                    # Insere os documentos no ChromaDB
                    st.write("Inserindo documentos no ChromaDB...")
                    inseridos, existentes = inserir_documentos(
                        documentos, 
                        uploaded_file.name
                    )

                    if inseridos > 0:
                        st.session_state.documento_atual = uploaded_file.name
                        st.session_state.processed_files.add(uploaded_file.name)
                        with result_placeholder:
                            st.success(f"""
                                Documento processado com sucesso!
                                - Nome: {uploaded_file.name}
                                - Chunks inseridos: {inseridos}
                                - Chunks com erro: {existentes}
                            """)
                    else:
                        with result_placeholder:
                            st.error(f"""
                                Não foi possível processar o documento.
                                - Erros encontrados: {existentes}
                                - Verifique o formato e conteúdo do arquivo.
                            """)

        except Exception as e:
            with result_placeholder:
                st.error(f"Erro ao processar o documento: {str(e)}")
            print(f"Erro detalhado: {e}")
        finally:
            set_processing_state(False)

# Mostra o documento atual e adiciona a capacidade de removê-lo
if st.session_state.documento_atual:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Documento atual: {st.session_state.documento_atual}")
    with col2:
        if st.button("Remover Documento", disabled=st.session_state.is_processing):
            st.session_state.documento_atual = None
            st.session_state.upload_state = None
            st.rerun()

