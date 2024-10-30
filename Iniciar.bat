cd .    # Este comando mantém você na pasta do projeto
call venv\Scripts\activate
start ollama serve
streamlit run app.py