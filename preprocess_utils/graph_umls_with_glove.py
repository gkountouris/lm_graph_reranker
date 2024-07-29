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
from elasticsearch import Elasticsearch

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

def elastic():
    ip = "http://localhost:9200"
    es = Elasticsearch([
            ip
            ],
            verify_certs=True,
            timeout=1000,
            max_retries=10,
            retry_on_timeout=True
        )
    return es

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def calculate_sparse_tensor_memory_usage_in_gb(x):

    final_indices, final_values = x
    # Calculate memory used by indices (assuming int64 by default, 8 bytes per element)
    indices_memory = final_indices[0].numel() * 8
    
    # Calculate memory used by values (assuming float32 by default, 4 bytes per element)
    values_memory = final_values.numel() * 4
    
    # Total memory usage in bytes
    total_memory_bytes = indices_memory + values_memory
    
    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (2**30)
    
    return total_memory_gb

def calculate_accuracy(predictions):
    # Count the number of True values (correct predictions) and divide by the length of the predictions list
    accuracy = sum(predictions) / len(predictions)
    return accuracy

def has_common_element(list1, list2):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    
    # Find the intersection
    common_elements = set1 & set2
    
    # Check if there is at least one common element
    return len(common_elements) > 0

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

def tf_loader_splited(folder):

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

        # torch_part = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(f'cuda:{idx}')
        torch_part = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(f'cpu')
        torch_tf_idf_parts.append(torch_part)
        
    del tf_idf_matrix 

    return torch_tf_idf_parts

def tf_loader(folder):
    # Load TF-IDF matrix
    tf_idf_matrix = load_npz(folder)

    # Convert the SciPy COO matrix to PyTorch sparse tensor
    tf_idf_matrix = tf_idf_matrix.tocoo()  # Ensure it is in COO format for easy conversion

    # Create indices array
    indices = torch.LongTensor(np.vstack((tf_idf_matrix.row, tf_idf_matrix.col)))

    # Create values array
    values = torch.FloatTensor(tf_idf_matrix.data)

    # Define shape of the sparse tensor
    shape = torch.Size(tf_idf_matrix.shape)

    # Create the PyTorch sparse tensor
    torch_tf_idf_matrix = torch.sparse.FloatTensor(indices, values, shape).to('cpu')  # Use .to('cuda:x') if GPU is needed

    return torch_tf_idf_matrix

def scipy_sparse_to_torch_sparse(matrix, device):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

def slice_sparse_tensor(row_indices, col_indices, sparse_tensor):
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Convert row_indices to tensor if it is not already one
    if not isinstance(row_indices, torch.Tensor):
        row_indices = torch.tensor(row_indices, device=indices.device)

    # Similarly, ensure col_indices is a tensor
    if not isinstance(col_indices, torch.Tensor):
        col_indices = torch.tensor(col_indices, device=indices.device)

    row_mask = torch.isin(indices[0], row_indices)
    col_mask = torch.isin(indices[1], col_indices)
    combined_mask = col_mask & row_mask
    final_indices = indices[:, combined_mask]
    final_values = values[combined_mask]

    return final_indices, final_values

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
    qc_ids, question, ans, data_id, bm_labels, bm_ids, scores, relevant_entities = data 

    cid2score = get_glove_score(bm_ids, question)
    
    bm_ids = set(bm_ids) - set(qc_ids)
    qa_nodes = set(qc_ids) | set(bm_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes

    return (sorted(qc_ids), sorted(bm_ids), question, bm_labels, ans, sorted(extra_nodes), data_id, scores)


def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2(data):
    qc_ids, bm_ids, question, bm_labels, ans, extra_nodes, data_id, scores = data
    cid2score = get_glove_score(qc_ids+bm_ids+extra_nodes, question)
    
    return (qc_ids, bm_ids, question, bm_labels, ans, extra_nodes, data_id, scores, cid2score)


def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3(data):
    qc_ids, bm_ids, question, bm_labels, ans, extra_nodes, data_id, scores, cid2score = data
    total_concepts = qc_ids + bm_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])
    schema_graph = total_concepts[:1000]
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    bmask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(bm_ids)))
    adj, concepts = concepts2adj(schema_graph)
    
    return adj, concepts, qmask, bmask, bm_labels, ans, data_id, scores, total_concepts


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

def generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, output_tensors_path, num_processes, train):

    print('#####################################################################################################')
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

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet, concept2name, counter, all_concepts
    counter = 0
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources()
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)
    
    # sparse_tensors = tf_loader_splited("data/pubmed/sparse_matrix.npz")

    all_concepts = set(range(297927))

    gc.collect()
    qa_data = []
    data_id_to_drop = []
    es = elastic()
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(tqdm(lines_ground)):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            obj = json.loads(lines_ground[j])
            QAcontext = "{}".format(obj['sent'])
            ans = dic.get('ans', ["1"])
            data_id = "{}".format(obj['id'])

            results = elastic_search_query.elastic_search_text(es, QAcontext, graph_entities=None, id_list=100, index_name = 'pubmed_documents')
            bm_labels = []
            scores = []
            bm_ids = set()

            for idx, res in enumerate(results['hits']['hits']):
                bm_labels.append(int(res['_id']))
                scores.append(res['_score'])
                for ent in res['_source']['graph_entities']:
                    bm_ids.add(ent)
            bm_ids = set(concept2id[c] for c in bm_ids)

            int_ans = []
            for an in ans:
                int_ans.append(int(an))

            # if (not common_member(int_ans, bm_labels)) and (train == True):
            #     data_id_to_drop.append(data_id)
            #     continue

            relevant_entities = set()
            for doc_id in ans:
                results = elastic_search_query.elastic_search_ID(es, doc_id, index_name = 'pubmed_documents')
                for ent in results['hits']['hits'][0]['_source']['graph_entities']:
                    relevant_entities.add(ent)
                
            relevant_entities = set(concept2id[c] for c in relevant_entities)

            if j == 100:
                break

            qa_data.append((q_ids, QAcontext, int_ans, data_id, bm_labels, bm_ids, scores, relevant_entities))
        fin_ground.close()

    print('Number of queries:', len(qa_data))

    q_ids_sum = 0
    bm_ids_sum = 0
    rel_ids_sum = 0
    for q_ids, QAcontext, int_ans, data_id, bm_labels, bm_ids, scores, relevant_entities in qa_data:
        q_ids_sum += len(q_ids)
        bm_ids_sum += len(bm_ids)
        rel_ids_sum += len(relevant_entities)

    print('Average number of nodes found in the query:', q_ids_sum/len(qa_data))
    print('Average number of nodes found in the documents:', bm_ids_sum/len(qa_data))
    print('Average number of relevant nodes found in the documents:', rel_ids_sum/len(qa_data))

    batch_size = 20
    num_processes = 20

    load_glove()
    # Process data in batches
    res1 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part1, qa_data, batch_size, num_processes, 'Part1')

    res2 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2, res1, batch_size, num_processes, 'Part2')

    # Clear memory
    global glove_w2v, id2glove
    del glove_w2v
    del id2glove

    from time import time

    start = time()
    res3 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3, res2, batch_size, num_processes, 'Part3')
    print("Time to construct graph = {} seconds".format((time()-start)/len(qa_data)))

    res3_adj_data = []    
    json_data = []  
    tensor_dicts = {}

    conc_sum = 0
    for adj, concepts, qmask, bmask, bm_labels, ans, data_id, scores, total_concepts in tqdm(res3):
        conc_sum += len(total_concepts)

    print('Average number of concepts found in the query:', conc_sum/len(qa_data))
    #     res3_adj_data.append((adj, concepts, qmask, bmask))
        
    #     # Assuming bm_labels, ans, and data_id are JSON-serializable as-is
    #     json_data.append({
    #         "bm_labels": bm_labels,
    #         "ans": ans,
    #         "data_id": data_id,
    #         "scores": scores
    #     })

    #     tensor_dicts[data_id] = []
    #     cols = [x+1 for x in concepts]
    #     for part_index in range(3):
    #         _indices, _values = slice_sparse_tensor(bm_labels, cols, sparse_tensors[part_index])
    #         tensor_dicts[data_id].append((_indices.to('cpu'), _values.to('cpu')))
    
    # combined_tensor_dict = {}
    # for data_id in tensor_dicts.keys():
    #     all_indices = []
    #     all_values = []

    #     # Iterate over each tuple in the list stored at tensor_dicts[data_id]
    #     for indices, values in tensor_dicts[data_id]:
    #         # Now, indices and values are directly accessible
    #         # Process or store these tensors as needed
    #         all_indices.append(indices)
    #         all_values.append(values)

    #     # Combine batches
    #     final_indices = torch.cat(all_indices, dim=1) if all_indices else torch.tensor([], dtype=torch.long)
    #     final_values = torch.cat(all_values, 0) if all_values else torch.tensor([], dtype=torch.float32)

    #     unique_labels, label_indices = torch.unique(final_indices[0], return_inverse=True)
    #     unique_cols, cols_indices = torch.unique(final_indices[1], return_inverse=True)

    #     combined_tensor_dict[data_id] = (final_indices, final_values)

    # # Save results to pickle file
    # os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    # with open(output_path, 'wb') as fout:
    #     pickle.dump(res3_adj_data, fout)

    # # Save bm_labels, ans, and data_id to a JSON file
    # json_output_path = output_path.replace('.pk', '.json')  # Assuming you want to keep the same base name
    # with open(json_output_path, 'w') as fjson:
    #     json.dump(json_data, fjson)

    # torch.save(combined_tensor_dict, output_tensors_path)
    # print(f'tensor data saved to {output_tensors_path}')
    # print()

    # updated_lines = [] 
    # with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
    #     lines_ground = fin_ground.readlines()
    #     for j, line in enumerate(tqdm(lines_ground)):
    #         dic = json.loads(line)
    #         if dic["id"] in data_id_to_drop:
    #             continue
    #         # Convert the dictionary back to a JSON string
    #         updated_line = json.dumps(dic)
    #         updated_lines.append(updated_line + '\n')  # Add newline for proper JSONL formatting
    #     fin_ground.close()
    
    # # Write the updated lines back to the file
    # with open(grounded_path, 'w', encoding='utf-8') as fout_ground:
    #     fout_ground.writelines(updated_lines)


    # # for adj, concepts, qmask, bmask, bm_labels, ans, data_id, scores in tqdm(res3):
    # #     res3_adj_data.append((adj, concepts, qmask, bmask))
        
    # #     # Assuming bm_labels, ans, and data_id are JSON-serializable as-is
    # #     json_data.append({
    # #         "bm_labels": bm_labels,
    # #         "ans": ans,
    # #         "data_id": data_id,
    # #         "scores": scores
    # #     })

    # #     tensor_dicts[data_id] = []
    # #     cols = [x+1 for x in concepts]
        
    # #     _indices, _values = slice_sparse_tensor(bm_labels, cols, sparse_tensors)
    # #     tensor_dicts[data_id].append((_indices.to('cpu'), _values.to('cpu')))