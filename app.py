import streamlit as st
from rag_main import configurar_llm, pesquisar_e_responder

st.set_page_config(page_title="Assistente RAG", page_icon="🤖", layout="wide")

# Inicialização do modelo
@st.cache_resource
def load_model():
    return configurar_llm()

llm = load_model()

st.title("Assistente RAG")

# Inicializa o histórico de chat se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "resultados" in message:
            with st.expander("Ver resultados da busca"):
                for i, resultado in enumerate(message["resultados"], 1):
                    st.write(f"Resultado {i}:")
                    st.json(resultado)

# Campo de entrada do usuário
if prompt := st.chat_input("O que você gostaria de saber?"):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simula a geração de resposta por partes
        resposta, resultados = pesquisar_e_responder(prompt, llm)
        
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

# Botão para limpar o histórico
if st.button("Limpar Conversa"):
    st.session_state.messages = []
    st._set_query_params()  # Força a recarga da página
