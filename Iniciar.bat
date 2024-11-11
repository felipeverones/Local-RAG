cd .
call venv\Scripts\activate
start ollama serve
streamlit run app.py