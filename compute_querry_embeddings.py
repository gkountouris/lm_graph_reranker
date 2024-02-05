import torch
from transformers import AutoTokenizer, AutoModel

def compute_embeddings(query, model_name_or_path="michiyasunaga/BioLinkBERT-large"):
    """
    Compute embeddings for a given query using BioLinkBERT.

    Args:
    - query (str): The text query for which to compute the embeddings.
    - model_name_or_path (str): The name or path of the BioLinkBERT model.

    Returns:
    - embeddings (torch.Tensor): The computed embeddings.
    """

    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # For sentence-level embeddings, you might want to average the token embeddings
    sentence_embeddings = embeddings.mean(dim=1)

    return sentence_embeddings

# Example usage
query = "Do DNA double-strand breaks play a causal role in carcinogenesis?"
embeddings = compute_embeddings(query)
print(embeddings)