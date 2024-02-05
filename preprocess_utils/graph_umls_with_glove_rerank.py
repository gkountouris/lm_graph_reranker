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
    qc_ids, question = data
    results = elastic_search_query.elastic_search_text(question, 100)
    bm_ids = set()

    for idx, res in enumerate(results['hits']['hits']):
        if idx < 10:
            for ent in res['_source']['graph_entities']:
                bm_ids.add(ent)

    bm_ids = set(concept2id[c] for c in bm_ids)
    bm_ids = bm_ids - qc_ids
    qa_nodes = set(qc_ids) | set(bm_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes

    return (sorted(qc_ids), sorted(bm_ids), question, sorted(extra_nodes))

def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part2(data):
    qc_ids, bm_ids, question, extra_nodes = data
    cid2score = get_glove_score(qc_ids+bm_ids+extra_nodes, question)
    
    return (qc_ids, bm_ids, question, extra_nodes, cid2score)

def concepts_to_adj_matrices_2hop_all_pair__use_glove__Part3(data):
    qc_ids, bm_ids, question, extra_nodes, cid2score = data
    schema_graph = qc_ids + bm_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])[:200] #score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    bmask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(bm_ids)))
    adj, concepts = concepts2adj(schema_graph)
    
    return adj, concepts, qmask, bmask


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

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            obj = json.loads(lines_ground[j])
            QAcontext = "{}".format(obj['sent'])
            qa_data.append((q_ids, QAcontext))
            
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


    # Save results
    os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    with open(output_path, 'wb') as fout:
        pickle.dump(res3, fout)

    print(f'adj data saved to {output_path}')
    print()
        