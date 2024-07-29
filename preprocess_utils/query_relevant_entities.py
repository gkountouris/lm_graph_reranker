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


def evaluate_concepts(relevant_entities, bm_ids):

    # Recall 
    recall = len(relevant_entities.intersection(bm_ids)) / len(relevant_entities) if relevant_entities else 0

    # Precision
    precision = len(relevant_entities.intersection(bm_ids)) / len(bm_ids) if bm_ids else 0

    # F1-Score 
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Jaccard Similarity
    jaccard_similarity = len(relevant_entities.intersection(bm_ids)) / len(relevant_entities.union(bm_ids))

    metric_sums = {
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1_score,
        'Jaccard Similarity': jaccard_similarity
    }

    return metric_sums


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
    qc_ids, question, data_id, bm_ids, relevant_ent = data 

    cid2score = get_glove_score(bm_ids, question)
    
    bm_ids_upgraded = set(bm_ids) - set(qc_ids)
    qa_nodes = set(qc_ids) | set(bm_ids_upgraded)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes

    return (sorted(qc_ids), sorted(bm_ids_upgraded), bm_ids, question, sorted(extra_nodes), data_id, relevant_ent)


def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2(data):
    qc_ids, bm_ids_upgraded, bm_ids, question, extra_nodes, data_id, relevant_ent = data
    cid2score = get_glove_score(qc_ids+bm_ids_upgraded+extra_nodes, question)
    
    return (qc_ids, bm_ids_upgraded, bm_ids, question, extra_nodes, data_id, cid2score, relevant_ent)


def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3(data):
    qc_ids, bm_ids_upgraded, bm_ids, question, extra_nodes, data_id, cid2score, relevant_ent = data
    schema_graph = qc_ids + bm_ids_upgraded + sorted(extra_nodes, key=lambda x: -cid2score[x])[:3000]
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    bmask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(bm_ids_upgraded)))
    adj, concepts = concepts2adj(schema_graph)
    
    relevant_ent = list(relevant_ent.intersection(set(concepts)))
    
    return adj, concepts, bm_ids, qmask, bmask, data_id, relevant_ent, qc_ids

#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

def find_relevant_entities(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet, concept2name, counter, all_concepts
    counter = 0
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources()
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    all_concepts = set(range(297927))

    gc.collect()
    num_queries = 0

    qa_data = []
    not_relevant_times = 0
    es = elastic()

    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(tqdm(lines_ground)):
            obj = json.loads(lines_ground[j])
            q_ids = set(concept2id[c] for c in obj['qc'])
            QAcontext = "{}".format(obj['sent'])
            ans = obj.get('ans', [])
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

            # if not common_member(int_ans, bm_labels) :
            #     continue

            relevant_entities = set()
            for doc_id in ans:
                results = elastic_search_query.elastic_search_ID(es, doc_id, index_name = 'pubmed_documents')
                for ent in results['hits']['hits'][0]['_source']['graph_entities']:
                    relevant_entities.add(ent)
                
            relevant_entities = set(concept2id[c] for c in relevant_entities)

            qa_data.append((q_ids, QAcontext, data_id, bm_ids, relevant_entities))
        fin_ground.close()

    batch_size = 100
    num_processes = 5

    load_glove()
    # Process data in batches
    res1 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part1, qa_data, batch_size, num_processes, 'Part1')

    res2 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2, res1, batch_size, num_processes, 'Part2')

    # Clear memory
    global glove_w2v, id2glove
    del glove_w2v
    del id2glove

    res3 = process_in_batches(concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3, res2, batch_size, num_processes, 'Part3')

    # mean_metrics_rele = {
    #     'Recall': 0,
    #     'Precision': 0,
    #     'F1-Score': 0,
    #     'Jaccard Similarity': 0
    # }

    # mean_metrics = {
    #     'Recall': 0,
    #     'Precision': 0,
    #     'F1-Score': 0,
    #     'Jaccard Similarity': 0
    # }
    json_data = []  

    for adj, concepts, bm_ids, qmask, bmask, data_id, relevant, qc_ids in tqdm(res3):

        # Assuming bm_labels, ans, and data_id are JSON-serializable as-is
        json_data.append({
            "relevant": relevant,
            "data_id": data_id,
        })

    #     metrics = evaluate_concepts(relevant, bm_ids)
    #     for key in mean_metrics_rele:
    #         mean_metrics_rele[key] += metrics[key]

    #     metrics = evaluate_concepts(relevant, set(concepts))
    #     for key in mean_metrics:
    #         mean_metrics[key] += metrics[key]

    # mean_metrics_rele = {key: value / len(qa_data) for key, value in mean_metrics_rele.items()}

    # print('Recall: This measures how many relevant entities are retrieved:', mean_metrics_rele['Recall'])
    # print('Precision: This measures how many of the retrieved entities are relevant:', mean_metrics_rele['Precision'])
    # print('F1-Score: This is the harmonic mean of Precision and Recall:', mean_metrics_rele['F1-Score'])
    # print('Jaccard Similarity: This measures the similarity between the retrieved concepts and the relevant concepts:', mean_metrics_rele['Jaccard Similarity'])
    # print('NO relevants :', not_relevant_times)

    # mean_metrics = {key: value / len(qa_data) for key, value in mean_metrics.items()}

    # print('Recall: This measures how many relevant entities are retrieved:', mean_metrics['Recall'])
    # print('Precision: This measures how many of the retrieved entities are relevant:', mean_metrics['Precision'])
    # print('F1-Score: This is the harmonic mean of Precision and Recall:', mean_metrics['F1-Score'])
    # print('Jaccard Similarity: This measures the similarity between the retrieved concepts and the relevant concepts:', mean_metrics['Jaccard Similarity'])
    # print('NO relevants :', not_relevant_times)

    # Save bm_labels, ans, and data_id to a JSON file
    
    with open(output_path, 'w') as fjson:
        json.dump(json_data, fjson)