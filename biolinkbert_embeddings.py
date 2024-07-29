from transformers import AutoTokenizer, AutoModel
import torch

import glob
import os
import json

from tqdm import tqdm

# Load the tokenizer and model
model_name = "michiyasunaga/BioLinkBERT-base"  # Adjust as necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# # Check if a GPU is available and move the model to GPU if it is
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

device = torch.device("cpu")


# Path to the directory containing the jsonl files
directory_path = 'data/pubmed_processed/statement/'

# Pattern to match all jsonl files in the directory
pattern = os.path.join(directory_path, '*.jsonl')

# List to store sentences
all_sents = []

# Iterate over all jsonl files in the directory

for i in tqdm(range(0, 1)):
    with open(directory_path + f'pubmed_eval_embeddings_{i:04d}.jsonl', 'r', encoding='utf-8') as file:
        # Iterate over each line (document) in the file
        for line in file:
            # Parse the JSON line to a Python dictionary
            doc = json.loads(line)
            # Extract the 'sent' field and add it to the list
            all_sents.append(doc['sent'])

# Now, all_sents contains all sentences extracted from the 'sent' field across all files
print(f'Total number of sentences extracted: {len(all_sents)}')

def process_sentences_in_batches(sentences, batch_size=32):
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)

# Assuming `all_sents` is a list of sentences
embeddings = process_sentences_in_batches(all_sents, batch_size=32)