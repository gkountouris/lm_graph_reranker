import argparse
import logging
import random
import shutil
import time
import json

# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb

from modeling import modeling_dragon
from utils import data_utils
from utils import optimization_utils
from utils import parser_utils
from utils import utils

import numpy as np

import socket, os, sys, subprocess
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import coo_matrix



logger = logging.getLogger(__name__)

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
        # Get the embeddings for the CLS token (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings


def theoretical_array_memory_size(num_rows, num_cols, dtype=np.float64):
    """
    Calculate the theoretical memory size for a NumPy array.
    """
    element_size = np.dtype(dtype).itemsize  # Size of one element in bytes
    num_elements = num_rows * num_cols       # Total number of elements in the array
    total_bytes = num_elements * element_size

    # Convert bytes to gigabytes
    total_gigabytes = total_bytes / (1024 ** 3)

    return total_gigabytes

def array_memory_size(array):
    """
    Calculate the memory size occupied by a NumPy array.
    """
    
    # Get number of elements in array
    num_elements = array.size
    
    # Get size of each element in bytes
    element_size = array.itemsize
    
    # Total memory in bytes
    total_bytes = num_elements * element_size
    
    # Convert bytes to kilobytes (1 KB = 1024 Bytes)
    total_kilobytes = total_bytes / 1024
    
    # Convert kilobytes to megabytes (1 MB = 1024 KB)
    total_megabytes = total_kilobytes / 1024

    # Convert megabytes to gigabytes (1 GB = 1024 MB)
    total_gigabytes = total_megabytes / 1024
    
    return total_gigabytes


def average_embeddings(embeddings):
    # Calculate the mean of the embeddings
    return np.mean(embeddings, axis=0)


def calculate_similarity(q_emb, d_emb):
    # Reshape embeddings to 2D array as required by cosine_similarity function
    q_emb = q_emb.reshape(1, -1)
    d_emb = d_emb.reshape(1, -1)
    
    # Calculate and return the cosine similarity score
    return cosine_similarity(q_emb, d_emb)[0][0]


def find_concept_in_file(filepath, concepts):
    line_numbers = []
    with open(filepath, 'r') as file:
        lines = file.read().splitlines()
        for concept in concepts:
            if concept in lines:
                line_number = lines.index(concept) + 1  # Adding 1 as list index starts from 0
                line_numbers.append(line_number)
            else:
                line_numbers.append(-1)
    return line_numbers


def entities_embeddings(cp_emb, contexts, entities):

    w_line_numbers = find_concept_in_file(contexts, entities)

    concept_embedding = []
    for line_number in w_line_numbers:
        if line_number != -1:
            concept_embedding.append(cp_emb[line_number - 1])  # Adjust for zero-based indexing

    return np.array(concept_embedding)


def weighted_similarities(data, vectors):

    weighted_vectors = np.zeros(1024)

    for tuples in data:

        # Retrieve the relevant vectors and multiply by weights
        selected_vectors = vectors[tuples[0]]
        weighted_vectors += selected_vectors * tuples[1]

    return weighted_vectors

if __name__ == "__main__":

    # Number of rows and columns
    num_rows = 23662580
    # num_rows = 15000/1000
    num_cols = 30
    # Calculate memory size
    memory_size_gb = theoretical_array_memory_size(num_rows, num_cols)
    print(f"Memory size in GB for a {num_rows}x{num_cols} array: {memory_size_gb} GB")

    # Given dense matrix
    dense_matrix = [
        [0, 0, 0, 2, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 4, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    ]

    # Convert to numpy array for easier processing
    dense_matrix = np.array(dense_matrix)

    print(array_memory_size(dense_matrix))

    # Extract non-zero elements and their indices
    row_indices, col_indices = dense_matrix.nonzero()
    print(row_indices)
    data = dense_matrix[row_indices, col_indices]

    print(array_memory_size(row_indices)+ array_memory_size(col_indices) + array_memory_size(data))

    # Create Sparse Matrix in COO format
    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(7, 10))

    # Print the sparse matrix
    print("Sparse Matrix in COO format:\n", sparse_matrix)

    # Optionally, convert to CSR format if you plan to do matrix operations
    sparse_matrix_csr = sparse_matrix.tocsr()

    loaded_data = np.load('/storage3/gkou/lm_graph/lm_graph/data/pubmed/data.npy', allow_pickle=True)
    # Convert to a dictionary if it's not already
    if isinstance(loaded_data, np.ndarray):
        loaded_data = loaded_data.item()


    print(loaded_data[0])


    # __spec__ = None

    # parser = parser_utils.get_parser()
    # args, _ = parser.parse_known_args()
    
    # kg = 'umls'
    # # Load pretrained concept embeddings
    # cp_emb = np.load('data/umls/ent_emb_blbertL.npy')
    # contexts = 'data/umls/concepts.txt'
    # loaded_data = np.load('/storage3/gkou/lm_graph/lm_graph/data/pubmed/data.npy', allow_pickle=True)
    # # Convert to a dictionary if it's not already
    # if isinstance(loaded_data, np.ndarray):
    #     loaded_data = loaded_data.item()

    # doc_embed = weighted_similarities(loaded_data['11178982'], cp_emb)

    # print(doc_embed)

    # memory_size = array_memory_size(cp_emb)
    # print("Memory size in GB:", memory_size)

    # # Example usage
    # query = "Do DNA double-strand breaks play a causal role in carcinogenesis?"
    # clc_embed = compute_embeddings(query)

    # print(clc_embed)

    # q_entities = ['C0596263']

    # concept_embedding = entities_embeddings(cp_emb, contexts, q_entities)

    # # # Calculate average embeddings
    # # # q_avg_emb = average_embeddings(q_entities)

    # # Calculate similarity
    # similarity_score = calculate_similarity(concept_embedding, clc_embed)
    # print(f"Similarity score: {similarity_score}")
    # # Calculate similarity
    # similarity_score = calculate_similarity(concept_embedding, doc_embed)
    # print(f"Similarity score: {similarity_score}")
    # # Calculate similarity
    # similarity_score = calculate_similarity(clc_embed, doc_embed)
    # print(f"Similarity score: {similarity_score}")

