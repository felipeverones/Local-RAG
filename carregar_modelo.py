from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import config

def carregar_modelo(nome_modelo):
    """Carrega o modelo de embeddings especificado.

    Args:
        nome_modelo: O nome do modelo a ser carregado.

    Returns:
        Uma tupla contendo o modelo de embeddings e o tokenizer (se aplicável).
    """
    if nome_modelo in [
        "distiluse-base-multilingual-cased-v2",
        "all-mpnet-base-v2",
        "all-MiniLM-L12-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]:
        model = SentenceTransformer(nome_modelo, device=config.DEVICE)
        return model, None
    else:
        model = AutoModel.from_pretrained(nome_modelo)  # Carregar o modelo
        tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
        device = torch.device(config.DEVICE)  # Converter para torch.device
        model.to(device)  # Configurar o dispositivo após o carregamento
        return model, tokenizer
