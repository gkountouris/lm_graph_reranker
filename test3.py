
import json
import sys
import math
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
from elastic import elastic_search_query
from elasticsearch import Elasticsearch

import pickle

import torch
import time
import socket, os, sys, subprocess
from utils import parser_utils


def Average(lst): 
    return sum(lst) / len(lst)

if __name__ == "__main__":

    pmid = 21645374
    q = "Biomolecular identification of allergenic pollen: a new perspective for aerobiological monitoring?"
    results = elastic_search_query.elastic_search_text(q, 1)
    ent = results['hits']['hits'][0]['_source']['graph_entities']
    print(ent)
    #{207946, 211909}
    #['C0003320', 'C0012854', 'C0205146', 'C0441621', 'C0443131', 'C0795585'] C0795585  C1445826 C1320237
    results = elastic_search_query.elastic_search_PMID(pmid)
    print(results)
    id = results['hits']['hits'][0]['_id']
    print("id", id)
    graph_entities = results['hits']['hits'][0]['_source']['graph_entities']
    print("graph_entities", graph_entities)
    text = results['hits']['hits'][0]['_source']['text']
    print("text", text)


    # Load Tf_Idf matrix
    tf_idf_matrix = load_npz("data/pubmed/sparse_matrix.npz")

    # Assuming tf_idf_matrix is your COO matrix
    # Print the shape of the matrix
    print("Shape of the matrix:", tf_idf_matrix.shape)

    # To see the row indices of non-zero elements
    print("Row indices:", tf_idf_matrix.row)

    # To see the column indices of non-zero elements
    print("Column indices:", tf_idf_matrix.col)

    # To see the non-zero values
    print("Data:", tf_idf_matrix.data)

    tf_idf_matrix = tf_idf_matrix.tocsc()
    # print(pmid_list)
    # selected_rows_matrix = tf_idf_matrix[int(id), :].tocoo()
    selected_rows_matrix = tf_idf_matrix[:, 19406].tocoo()

    print("Shape of the selected matrix:", selected_rows_matrix.shape)

    # To see the row indices of non-zero elements
    print("Row indices:", selected_rows_matrix.row, len(selected_rows_matrix.row))

    # To see the column indices of non-zero elements
    print("Column indices:", selected_rows_matrix.col, len(selected_rows_matrix.col))

    # To see the non-zero values
    print("Data:", selected_rows_matrix.data, len(selected_rows_matrix.data))

    # Find the index
    index = np.where(selected_rows_matrix.row == int(id))

    # To see the non-zero values
    print("Max Data:", max(selected_rows_matrix.data))

    # To see the non-zero values
    print("Average Data:", Average(selected_rows_matrix.data))

    # To see the non-zero values
    print("Data:", selected_rows_matrix.data[index])

    # print(selected_rows_matrix)
