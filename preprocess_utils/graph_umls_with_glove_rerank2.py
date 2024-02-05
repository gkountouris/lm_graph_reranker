import os
import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
# from .conceptnet import merged_relations
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool, Manager
from collections import OrderedDict
from elastic import elastic_search_query

from scipy.sparse import load_npz
from .maths import *
import gc


concept2id = None
concept2name = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_in_batches(function, data, batch_size, num_processes, desc):
    results = []
    for batch in tqdm(batch_data(data, batch_size), total=len(data) // batch_size, desc=desc):
        with Pool(num_processes) as p:
            batch_results = list(p.imap(function, batch))
        results.extend(batch_results)
    return results

def split_matrix(matrix, parts=3):
    rows, cols = matrix.shape
    rows_per_part = rows // parts
    matrices = []

    for i in range(parts):
        # Define row start and end for slicing
        row_start = rows_per_part * i
        row_end = rows_per_part * (i + 1) if i < parts - 1 else rows

        # Find the indices that belong to the current part
        part_indices = (matrix.row >= row_start) & (matrix.row < row_end)

        # Create new COO matrix for the current part
        part_matrix = coo_matrix((matrix.data[part_indices], 
                                  (matrix.row[part_indices] - row_start, matrix.col[part_indices])),
                                 shape=(row_end - row_start, cols))

        matrices.append(part_matrix)

    return matrices

def tf_loader(folder):

    # Load Tf_Idf matrix
    tf_idf_matrix = load_npz(folder)

    tf_idf_matrix = tf_idf_matrix.tocsr()

    num_devices = 3

    # Split the matrix into parts equal to the number of devices
    split_indices = np.linspace(0, tf_idf_matrix.shape[0], (num_devices + 1), dtype=int)
    
    torch_tf_idf_parts = []
    num_parts = len(split_indices) - 1  # Total number of parts to be distributed

    # Iterate over the number of parts
    for idx in range(num_parts):
        start, end = split_indices[idx], split_indices[idx + 1]
        part = tf_idf_matrix[start:end, :].tocoo()

        # Adjust row indices to reflect original position
        adjusted_row_indices = part.row + start

        # Create PyTorch sparse tensor with adjusted indices
        i = torch.LongTensor(np.vstack((adjusted_row_indices, part.col)))
        v = torch.FloatTensor(part.data)
        shape = (tf_idf_matrix.shape[0], part.shape[1])  # Keep original number of rows

        torch_part = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(f'cuda:{idx}')
        torch_tf_idf_parts.append(torch_part)
        
    del tf_idf_matrix 

    return torch_tf_idf_parts


def scipy_sparse_to_torch_sparse(matrix, device):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)


def slice_sparse_tensor(row_indices, col_indices, sparse_tensor, batch_size=20):
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Convert row_indices to tensor if it is not
    if not isinstance(row_indices, torch.Tensor):
        row_indices = torch.tensor(row_indices, device=indices.device)

    # The shape of the original tensor
    shape = sparse_tensor.shape

    # Initialize lists to store batches of indices and values
    batch_indices = []
    batch_values = []

    # Process col_indices in batches
    for i in range(0, len(col_indices), batch_size):
        batch_col_indices = col_indices[i:i + batch_size]
        if not isinstance(batch_col_indices, torch.Tensor):
            batch_col_indices = torch.tensor(batch_col_indices, device=indices.device)

        # Apply the mask for columns in this batch
        col_mask = torch.isin(indices[1], batch_col_indices)
        col_filtered_indices = indices[:, col_mask]
        col_filtered_values = values[col_mask]

        # Apply the mask for rows
        row_mask = torch.isin(col_filtered_indices[0], row_indices)
        final_indices = col_filtered_indices[:, row_mask]
        final_values = col_filtered_values[row_mask]

        # Add the batch indices and values to the lists
        batch_indices.append(final_indices)
        batch_values.append(final_values)

    # Concatenate all batches
    final_indices = torch.cat(batch_indices, dim=1) if batch_indices else torch.tensor([], dtype=torch.long)
    final_values = torch.cat(batch_values, dim=0) if batch_values else torch.tensor([], dtype=values.dtype)

    # Create the new sparse tensor using the concatenated indices and values
    selected_tensor = torch.sparse_coo_tensor(final_indices, final_values, torch.Size(shape))

    return selected_tensor


def load_resources():
    global concept2id, id2concept, relation2id, id2relation, concept2name
    id2concept = [w.strip() for w in open('data/umls/concepts.txt')]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    concept2name = {}
    for line in open('data/umls/concept_names.txt'):
        c, n = line.strip().split('\t')
        concept2name[c] = n
    id2relation = [r.strip() for r in open('data/umls/relations.txt')]
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    import pickle
    with open(cpnet_graph_path, 'rb') as f:
        cpnet = pickle.load(f)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    if n_node == 0:
        print('subgraph with no node')
        adj = coo_matrix(np.zeros((n_rel * n_node, n_node), dtype=np.uint8))
    else:
        adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


######################################################################
import re

token_pattern = re.compile(r"(?u)\b\w+\b")

glove_w2v = None
id2glove = None

def load_glove():
    global glove_w2v, id2glove

    print ('Loading glove...')
    glove_w2v = {}
    for line in tqdm(open('data/glove/glove.6B/glove.6B.50d.txt')):
        elms = line.split()
        glove_w2v[elms[0]] = np.array(elms[1:], dtype=float)
    print ('Loaded glove.')

    print ('Mapping concepts to glove vecs...')
    global concept2id, id2concept, relation2id, id2relation, concept2name
    if concept2id is None:
        load_resources()
    id2glove = []
    for id, concept in enumerate(tqdm(id2concept)):
        name = concept2name[concept]
        name = name.replace('_', ' ')
        id2glove.append(sent2glove(name))
    print ('Mapped concepts to glove vecs.')

def sent2glove(sent):
    words = token_pattern.findall(sent.lower())
    vec = np.sum([glove_w2v.get(w, np.zeros((50,), dtype=float)) for w in words], axis=0)
    if not isinstance(vec, np.ndarray):
        vec = np.zeros((50,), dtype=float)
    l2norm = np.sqrt((vec **2).sum())
    vec = vec / (l2norm +1e-8)
    return vec

def get_glove_score(cids, question):
    if len(cids) == 0:
        return {}
    sent_vec = sent2glove(question) #[dim,]
    concept_vecs = np.stack([id2glove[cid] for cid in cids]) #[nodes, dim]
    scores = list(concept_vecs.dot(sent_vec)) #[nodes,]
    assert len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1])) #score: from high to low
    return cid2score

def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part1(data):
    qc_ids, question, ans, id = data
    results = elastic_search_query.elastic_search_text(question, 100)
    bm_ids = set()
    bm_lables = []

    for idx, res in enumerate(results['hits']['hits']):
        bm_lables.append(int(res['_id']))
        if idx < 10:
            for ent in res['_source']['graph_entities']:
                bm_ids.add(ent)

    if int(ans) not in bm_lables:
        bm_lables[-1] = int(ans)

    bm_ids = set(concept2id[c] for c in bm_ids)
    bm_ids = bm_ids - qc_ids
    qa_nodes = set(qc_ids) | set(bm_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes

    return (sorted(qc_ids), sorted(bm_ids), question, bm_lables, sorted(extra_nodes), id)

def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2(data):
    qc_ids, bm_ids, question, bm_lables, extra_nodes, id = data
    cid2score = get_glove_score(qc_ids+bm_ids+extra_nodes, question)
    
    return (qc_ids, bm_ids, question, bm_lables, extra_nodes, id, cid2score)

def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3(data):
    qc_ids, bm_ids, question, bm_lables, extra_nodes, id, cid2score = data
    schema_graph = qc_ids + bm_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])[:200] #score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    bmask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(bm_ids)))
    adj, concepts = concepts2adj(schema_graph)
    
    return adj, concepts, qmask, bmask, bm_lables, id


#####################################################################################################


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

def generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, output_tensors_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet, concept2name, counter
    counter = 0
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources()
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)
    
    sparse_tensors = tf_loader("data/pubmed/sparse_matrix.npz")

    gc.collect()
    qa_data = []
    ans_dict = {}
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            obj = json.loads(lines_ground[j])
            QAcontext = "{}".format(obj['sent'])
            ans = "{}".format(obj['ans'])
            id = "{}".format(obj['id'])
            qa_data.append((q_ids, QAcontext, ans, id))
            ans_dict[obj['id']] = obj['ans']
            
    batch_size = 100
    num_processes = 5

    # Process data in batches
    res1 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part1, qa_data, batch_size, num_processes, 'Part1')

    load_glove()
    res2 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2, res1, batch_size, num_processes, 'Part2')

    # Clear memory
    global glove_w2v, id2glove
    del glove_w2v
    del id2glove

    res3 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3, res2, batch_size, num_processes, 'Part3')

    res3_adj_data = []
    tensor_dicts = {}

    # Assuming the tensors are stored in a list called sparse_tensors
    row_starts = [sparse_tensors[i]._indices()[0, 0].item() for i in range(3)]
    row_ends = row_starts[1:] + [sparse_tensors[-1].shape[0]]

    for adj, concepts, qmask, bmask, bm_lables, id in tqdm(res3):
        tensor_dicts[id] = []
        res3_adj_data.append((adj, concepts, qmask, bmask))
        cols = [x+1 for x in concepts[:400]]
        try:
            for part_index in range(3):
                row_start = row_starts[part_index]
                row_end = row_ends[part_index]

                part_bm_labels = [label for label in bm_lables if row_start <= label < row_end]

                if part_bm_labels:
                    sliced_sparse_tensor = slice_sparse_tensor(part_bm_labels, cols, sparse_tensors[part_index], batch_size=20)
                    tensor_dicts[id].append(sliced_sparse_tensor.to('cpu'))
        except:
            print("problem", id)


    combined_tensor_dict = {}
    for id in tensor_dicts.keys():
        all_indices = []
        all_values = []

        for sparse_matrix in tensor_dicts[id]:
            # Extract indices and values
            indices = sparse_matrix._indices().clone()
            values = sparse_matrix._values()

            all_indices.append(indices)
            all_values.append(values)
            shape = sparse_matrix.shape

        # Combine batches
        final_indices = torch.cat(all_indices, dim=1) if all_indices else torch.tensor([], dtype=torch.long)
        final_values = torch.cat(all_values, 0) if all_values else torch.tensor([], dtype=torch.float32)
        combined_tensor_dict[id] = torch.sparse_coo_tensor(final_indices, final_values, torch.Size(shape))

            
    # Save results 183 418 
    os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    with open(output_path, 'wb') as fout:
        pickle.dump(res3_adj_data, fout)

    print(f'adj data saved to {output_path}')
    print()

    torch.save(combined_tensor_dict, output_tensors_path)
    print(f'tensor data saved to {output_tensors_path}')
    print()

