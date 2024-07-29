
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

def get_devices(args):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""

    if args.local_rank == -1 or not args.cuda:
        if torch.cuda.device_count() >= 3 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            device2 = torch.device("cuda:2")  # Add third device
            print("device0: {}, device1: {}, device2: {}".format(device0, device1, device2))
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            print("device0: {}, device1: {}".format(device0, device1))
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device0 = torch.device("cuda", args.local_rank)
        device1 = device0
        torch.distributed.init_process_group(backend="nccl")

    args.world_size = world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    print ("Process rank: %s, device: %s, distributed training: %s, world_size: %s" %
              (args.local_rank,
              device0,
              bool(args.local_rank != -1),
              world_size), file=sys.stderr)

    return device0, device1, device2 if 'device2' in locals() else device0

def size_of(docs):

    size_in_bytes = sys.getsizeof(docs)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024

    print(f"Size in bytes: {size_in_bytes}")
    print(f"Size in KB: {size_in_kb}")
    print(f"Size in MB: {size_in_mb}")

def weight_similarities(tuples_list, similarity_scores):

    # Process each list of tuples
    for tuples in tuples_list:
        weighted_sums = np.zeros(len(similarity_scores))
        for index, weight in tuples:
            weighted_sums += similarity_scores[index] * weight
    return weighted_sums

def load_concepts(filepath):
    concepts = {}
    with open(filepath, 'r') as file:
        for i, line in enumerate(file, start=1):
            concepts[line.strip()] = i
    return concepts

def index_pubmed_docs(pubmed_path):

    total_files = 1167  # Total number of files to process

    # Dictionary to hold PMID to index mapping
    pmid_to_index = {}

    index = 1  # Start indexing from 1
    for i in tqdm(range(1, total_files)):
        with open(pubmed_path + f"/pubmed23n{i:04}.json", "r") as f:
            docs = json.load(f)
            for doc in docs:
                PMID = doc['PMID']
                if PMID not in pmid_to_index:  # Check if PMID is already indexed
                    pmid_to_index[PMID] = index
                    index += 1  # Increment index for next PMID
    with open(pubmed_path + '/pmid_to_index.pkl', 'wb') as f:
        pickle.dump(pmid_to_index, f)

def tf_idf_score(docs, concepts):

    doc_entities = {doc['PMID']: set(doc['graph_entities']) for doc in docs}
    entity_doc_count = defaultdict(int)
    for entities in doc_entities.values():
        for entity in entities:
            entity_doc_count[entity] += 1

    total_docs = len(doc_entities)
    idf_scores = {entity: math.log(total_docs / count, 10) for entity, count in entity_doc_count.items()}

    # tf_idf_scores = {}
    index = 0
    rows_list = []
    col_list = []
    data_list = []

    for pmid, entities in tqdm(doc_entities.items()):
        doc_scores = {entity: (1 / len(entities)) * idf_scores.get(entity, 0) for entity in entities}
        sum_scores = sum(doc_scores.values())

        for entity, score in doc_scores.items():
            rows_list.append(index)
            col_list.append(concepts.get(entity, -1))
            data_list.append(score / sum_scores)

        doc_scores = {concepts.get(entity, -1): score / sum_scores for entity, score in doc_scores.items()}
        # doc_tfidf_tuples = [(concepts.get(entity, -1), score) for entity, score in doc_scores.items()]
        # tf_idf_scores[pmid] = doc_tfidf_tuples
        # rows
        index += 1

    print(index)
    row  = np.array(rows_list)
    col  = np.array(col_list)
    data = np.array(data_list)
    sparse_doc_entity_matrix = coo_matrix((data, (row, col)))

    # print(tf_idf_score)
    # # Assuming 'data' is your list of tuples
    # array_data = np.array(tf_idf_scores)
    # print(array_data)

    return sparse_doc_entity_matrix


# def sparse_matrix_creation(docs, concepts):

#     doc_entities = {doc['_id']: set(doc['graph_entities']) for doc in docs}
#     entity_doc_count = defaultdict(int)
#     for entities in doc_entities.values():
#         for entity in entities:
#             entity_doc_count[entity] += 1

#     total_docs = len(doc_entities)
#     idf_scores = {entity: math.log(total_docs / count, 10) for entity, count in entity_doc_count.items()}

#     rows_list = []
#     col_list = []
#     data_list = []
#     for _id, entities in tqdm(doc_entities.items()):
#         doc_scores = {entity: (1 / len(entities)) * idf_scores.get(entity, 0) for entity in entities}
#         sum_scores = sum(doc_scores.values())
#         for entity, score in doc_scores.items():
#             rows_list.append(int(_id))
#             col_list.append(concepts.get(entity, -1))
#             data_list.append(score / sum_scores)

#     row  = np.array(rows_list)
#     col  = np.array(col_list)
#     data = np.array(data_list)
    
#     sparse_doc_entity_matrix = coo_matrix((data, (row, col)))
#     return sparse_doc_entity_matrix

def batch_query_elasticsearch_with_scroll(pmids):

    # Initialize Elasticsearch client
    index_name = 'pubmed_documents'
    ip="http://localhost:9200"
    es = Elasticsearch([
            ip
            ],
                verify_certs=True,
                timeout=1000,
                max_retries=10,
                retry_on_timeout=True
            )
    
    query = {
        "query": {
            "terms": {
                "PMID": pmids
            }
        }
    }

    # Initialize the scroll
    scroll = '2m'  # Keep the scroll window open for 2 minutes
    response = es.search(index=index_name, body=query, scroll=scroll, size=1000)

    # Keep track of the batch results
    results = []

    while True:
        # Get the scroll ID
        scroll_id = response['_scroll_id']

        # Get the batch of results
        results.extend(response['hits']['hits'])

        # Fetch the next batch
        response = es.scroll(scroll_id=scroll_id, scroll=scroll)

        # Check if we have reached the end of the results
        if not response['hits']['hits']:
            break

    return results

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

    device = get_devices(args)

    pubmed_path = './data/pubmed_processed'
    concept_path = '/storage3/gkou/lm_graph/lm_graph/data/umls/concepts.txt'
    output_file = '/storage3/gkou/lm_graph/lm_graph/data/pubmed'

    concepts = load_concepts(concept_path)

    total_files = 1167  # Total number of files to process 1167
    
    filtered_docs = []

    for i in tqdm(range(1, total_files)):
        with open(pubmed_path + f"/pubmed23n{i:04}.json", "r") as f:
            docs = json.load(f)
        
        # Collect PMIDs
        pmids = [doc['PMID'] for doc in docs if 'PMID' in doc and 'graph_entities' in doc]

        # Batch query Elasticsearch
        response = batch_query_elasticsearch_with_scroll(pmids)

        # Process the response documents
        for doc in docs:
            # Find the corresponding Elasticsearch response for each doc
            es_responses = [hit for hit in response if hit['_source']['PMID'] == doc['PMID']]

            # Process each response
            for es_response in es_responses:
                if es_response:  # Check if the response is valid
                    index = es_response['_id']
                    filtered_docs.append({'_id': int(index), 'PMID': doc['PMID'], 'graph_entities': doc['graph_entities']})

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
    save_npz(output_file + '/sparse_matrix2.npz', sparse_matrix)


    # # np.save(output_file + '/data.npy', tf_idf_scores)

    # # loaded_data = np.load(output_file + '/data.npy', allow_pickle=True)

    # # sparse_matrix = load_npz(output_file +'/sparse_matrix.npz')

    # # Convert to COO format (needed for PyTorch conversion)
    # sparse_matrix_coo = sparse_matrix.tocoo()

    # # Create the values, row indices, and column indices tensors
    # values = torch.FloatTensor(sparse_matrix_coo.data)
    # indices = torch.LongTensor([sparse_matrix_coo.row, sparse_matrix_coo.col])

    # # Create a PyTorch sparse tensor
    # sparse_matrix_torch = torch.sparse.FloatTensor(indices, values, torch.Size(sparse_matrix_coo.shape))
    # print(torch.Size(sparse_matrix_coo.shape))

    # # Move sparse_matrix_torch to the desired device (e.g., GPU)
    # sparse_matrix_torch = sparse_matrix_torch.to(device[1])

    # # Define and prepare S matrix
    # E = sparse_matrix_torch.shape[1]  # Adjust E to match the shape of your sparse matrix
    # S = np.random.randn(E, 1)
    # S = torch.from_numpy(S).to_sparse()
    # S = S.to(torch.float32).to(device[1])

    # # Time measurement start
    # s_time = time.time()
    # # Matrix multiplication
    # result = torch.sparse.mm(sparse_matrix_torch, S)

    # print(result)
    # # Time measurement end
    # e_time = time.time()
    # print("Seconds =", e_time - s_time)
