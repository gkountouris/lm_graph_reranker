
import json
import sys
import math
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
# from elastic import elastic_search_query
from elasticsearch import Elasticsearch

import pickle

import torch
import time
import socket, os, sys, subprocess
from utils import parser_utils


def load_concepts(filepath):
    concepts = {}
    with open(filepath, 'r') as file:
        for i, line in enumerate(file, start=1):
            concepts[line.strip()] = i
    return concepts

def sparse_matrix_creation(docs, concepts, idf_scores):
    rows_list = []
    col_list = []
    data_list = []
    for doc in docs:
        entities = set(doc['graph_entities'])
        doc_scores = {entity: (1 / len(entities)) * idf_scores.get(entity, 0) for entity in entities}
        sum_scores = sum(doc_scores.values())
        for entity, score in doc_scores.items():
            rows_list.append(doc['_id'])
            col_list.append(concepts.get(entity, -1))
            data_list.append(score / sum_scores)
    row  = np.array(rows_list)
    col  = np.array(col_list)
    data = np.array(data_list)
    
    return coo_matrix((data, (row, col)))

if __name__ == "__main__":

    __spec__ = None

    parser = parser_utils.get_parser()

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")

    args, _ = parser.parse_known_args()


    pubmed_path = './data/pubmed_processed'
    concept_path = '/storage3/gkou/lm_graph/lm_graph/data/umls/concepts.txt'
    output_file = '/storage3/gkou/lm_graph/lm_graph/data/pubmed'

    concepts = load_concepts(concept_path)

    total_files = 1167  # Total number of files to process 1167
    
    filtered_docs = []
    index = 0
    for i in tqdm(range(1, total_files)):
        with open(pubmed_path + f"/pubmed23n{i:04}.json", "r") as f:
            docs = json.load(f)
        # Process the response documents
        for doc in docs:
            index += 1
            filtered_docs.append({'_id': index, 'graph_entities': doc['graph_entities']})

    # Calculate total number of documents based on filtered documents
    total_docs = len(filtered_docs)  # Total number of filtered documents

    # Calculate entity document counts from filtered_docs
    entity_doc_count = defaultdict(int)
    for doc in filtered_docs:  # Use filtered_docs here
        entities = set(doc['graph_entities'])
        for entity in entities:
            entity_doc_count[entity] += 1

    # Compute IDF scores once, outside the loop
    idf_scores = {entity: math.log(total_docs / count, 10) for entity, count in entity_doc_count.items()}

    # Call the sparse_matrix_creation function with precomputed IDF scores
    sparse_matrix = sparse_matrix_creation(filtered_docs, concepts, idf_scores)

    # Save the created sparse matrix
    save_npz(output_file + '/sparse_matrix3.npz', sparse_matrix)

