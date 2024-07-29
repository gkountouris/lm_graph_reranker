import argparse
from multiprocessing import cpu_count
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.graph_umls_with_glove_retrieval2 import generate_adj_data_from_grounded_concepts_umls_retrieval__use_glove
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

from collections import Counter

from scipy.sparse import load_npz
import gc
import statistics

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

def calculate_percentage(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    # Find common entities between the two sets
    intersection = set1.intersection(set2)
    
    # Calculate the percentage
    if len(set2) == 0:  # To avoid division by zero if the first list is empty
        return 0.0
    percentage = (len(intersection) / len(set2)) * 100
    
    return percentage

def calculate_recall_at_k(bm_labels, ans, k):
    if not ans:  # If there are no relevant documents, avoid division by zero
        return 0.0

    top_k = bm_labels[:k]  # Consider only the top k documents
    retrieved_relevant_documents = sum(1 for doc in top_k if doc in ans)
    total_relevant_documents = len(ans)
    
    recall = retrieved_relevant_documents / total_relevant_documents
    return recall

def calculate_recall(bm_labels, ans):
    if not ans:  # If there are no relevant documents, avoid division by zero
        return 0.0
    
    retrieved_relevant_documents = set(bm_labels) & set(ans)
    recall = len(retrieved_relevant_documents) / len(ans)
    return recall

def calculate_ap(bm_labels, ans, k):
    if not ans:  # If there are no relevant documents, return 0.0 to avoid division by zero
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, label in enumerate(bm_labels[:k]):
        if label in ans:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    average_precision = sum_precisions / len(ans)
    return average_precision

def calculate_mrr_at_k(bm_labels, ans, k):
    for i, label in enumerate(bm_labels[:k]):
        if label in ans:
            return 1.0 / (i + 1)  # Return the reciprocal of the rank (1-based index)
    return 0.0  # Return 0 if no relevant documents are found within the top k

def calculate_mean_metric(predictions):
    # Count the number of True values (correct predictions) and divide by the length of the predictions list
    accuracy = sum(predictions) / len(predictions)
    return accuracy

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


def check_bm25_entities(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, output_tensors_path, num_processes):

    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet, concept2name, counter
    counter = 0
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources()
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    adj_concept_pairs = []
    print(f'Loading {output_path}...')
    with open(output_path, "rb") as in_file:
        try:
            while True:
                ex = pickle.load(in_file)
                if type(ex) == dict:
                    adj_concept_pairs.append(ex)
                elif type(ex) == tuple:
                    adj_concept_pairs.append(ex)
                elif type(ex) == list:
                    assert len(ex) > 10
                    adj_concept_pairs.extend(ex)
                else:
                    raise TypeError("Invalid type for ex.")
        except EOFError:
            pass

    q_concepts = []
    f = open('data/umls/concepts_positions.json', 'r')
    positions = json.load(f)
    for j, adj in enumerate(adj_concept_pairs):
        concepts = adj[1]+1
        concept_data = []
        for index in concepts:  # Iterate over each index in the list/array
            concept_data.append(positions[str(index)])
        q_concepts.append(concept_data)
        

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            # dic = json.loads(line)
            # q_ids = set(concept2id[c] for c in dic['qc'])
            obj = json.loads(lines_ground[j])
            QAcontext = "{}".format(obj['sent'])
            ans = obj['ans']
            data_id = "{}".format(obj['id'])
            qa_data.append((QAcontext, ans, data_id))

    rec = []
    perc = []
    reck = []
    ap10 = []
    ap5 = []
    mrr10 = [] 
    mrr5 = []
    avg_conc = []
    bm_max_ = []
    extra_nodes_max_ = []
    bm_avg_len = 0
    average_connected = 0
    extra_nodes_avg_len = 0


    for q_data, enti in tqdm(zip(qa_data, q_concepts)):

        question, ans, doc_id = q_data

        results = elastic_search_query.elastic_search_text(question, graph_entities=None, id_list=2000)
        bm_labels = []
        bm_ids = set()
        for idx, res in enumerate(results['hits']['hits']):
            bm_labels.append(int(res['_id']))
            for ent in res['_source']['graph_entities']:
                bm_ids.add(ent)
        bm_ids = set(concept2id[c] for c in bm_ids)

        extra_nodes = set()
        for qid in bm_ids:
            for aid in bm_ids:
                if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                    extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])

        int_ans = []
        for an in ans:
            int_ans.append(int(an))

        rec.append(calculate_recall(bm_labels, int_ans))
        reck.append(calculate_recall_at_k(bm_labels, int_ans, 100))
        ap10.append(calculate_ap(bm_labels, int_ans, 10))
        ap5.append(calculate_ap(bm_labels, int_ans, 5))
        mrr10.append(calculate_mrr_at_k(bm_labels, int_ans, 10))
        mrr5.append(calculate_mrr_at_k(bm_labels, int_ans, 5))

    accuracy = calculate_mean_metric(perc)
    print(f"Entities mean percentage: {accuracy:.2f}")
    accuracy = calculate_mean_metric(rec)
    print(f"Recall: {accuracy:.2f}")
    accuracy = calculate_mean_metric(reck)
    print(f"Recall at 100: {accuracy:.2f}")
    accuracy = calculate_mean_metric(ap10)
    print(f"MAP at 10: {accuracy:.2f}")
    accuracy = calculate_mean_metric(ap5)
    print(f"MAP at 5: {accuracy:.2f}")
    accuracy = calculate_mean_metric(mrr10)
    print(f"MRR at 10: {accuracy:.2f}")
    accuracy = calculate_mean_metric(mrr5)
    print(f"MRR at 5: {accuracy:.2f}")


def check_bm25_questions(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, output_tensors_path, num_processes):

    es = elastic()

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in tqdm(enumerate(lines_ground)):
            obj = json.loads(lines_ground[j])
            q_ids = set(obj['qc'])
            QAcontext = "{}".format(obj['sent'])
            ans = obj['ans']
            data_id = "{}".format(obj['id'])
            qa_data.append((QAcontext, ans, data_id, q_ids))

    entities_by_query = []

    rec = []
    reck = []
    ap10 = []
    ap5 = []
    mrr10 = []
    mrr5 = []
    percentage_unique_avg = []

    for question, ans, doc_id, q_ids in tqdm(qa_data):

        int_ans = []
        for an in ans:
            int_ans.append(int(an))

        bm_ids_ans_list_not = list()
        bm_ids_ans = set()
        bm_ids = set()
        bm_ids_list = list()
        bm_ids_ans_list = list()

        # results = elastic_search_query.elastic_search_text(es, QAcontext, graph_entities=q_ids, id_list=50, index_name='pubmed_documents')
        results = elastic_search_query.elastic_search_text(es, question, graph_entities=None, id_list=4000, index_name = 'pubmed_documents')
        bm_labels = []
        entities_idx = []
        entities_idx_ans = []
        for idx, res in enumerate(results['hits']['hits']):
            bm_labels.append(int(res['_id']))
            entities = set()
            for ent in res['_source']['graph_entities']:
                bm_ids.add(ent)
                bm_ids_list.append(ent)
                entities.add(ent)
            if int(res['_id']) not in int_ans:
                entities_idx.append(entities)
            if int(res['_id']) in int_ans:
                entities_idx_ans.append(entities)

        rec.append(calculate_recall(bm_labels, int_ans))
        reck.append(calculate_recall_at_k(bm_labels, int_ans, 100))
        ap10.append(calculate_ap(bm_labels, int_ans, 10))
        ap5.append(calculate_ap(bm_labels, int_ans, 5))
        mrr10.append(calculate_mrr_at_k(bm_labels, int_ans, 10))
        mrr5.append(calculate_mrr_at_k(bm_labels, int_ans, 5))

    # print(f"percentage_unique_avg: {calculate_mean_metric(percentage_unique_avg):.2f}%")
    accuracy = calculate_mean_metric(rec)
    print(f"Recall: {accuracy:.2f}")
    accuracy = calculate_mean_metric(reck)
    print(f"Recall at 100: {accuracy:.2f}")
    accuracy = calculate_mean_metric(ap10)
    print(f"MAP at 10: {accuracy:.2f}")
    accuracy = calculate_mean_metric(ap5)
    print(f"MAP at 5: {accuracy:.2f}")
    accuracy = calculate_mean_metric(mrr10)
    print(f"MRR at 10: {accuracy:.2f}")
    accuracy = calculate_mean_metric(mrr5)
    print(f"MRR at 5: {accuracy:.2f}")    
    
output_paths = {
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
}

for dname in ['BioASQ']:
    output_paths[dname] = {
        'statement': {
            'train':  f'./data/{dname}/statement/training11b.statement.jsonl',
            'dev':  f'./data/{dname}/statement/dev11b.statement.jsonl',
            'test1':  f'./data/{dname}/statement/11B1_golden.statement.jsonl',
            'test2':  f'./data/{dname}/statement/11B2_golden.statement.jsonl',
            'test3':  f'./data/{dname}/statement/11B3_golden.statement.jsonl',
            'test4':  f'./data/{dname}/statement/11B4_golden.statement.jsonl',
        },
        'graph': {
            'adj-train':  f'./data/{dname}/graph/training11b.graph.adj.pk',
            'adj-dev':  f'./data/{dname}/graph/dev11b_bm25.graph.adj.pk',
            'adj-test1':  f'./data/{dname}/graph/11B1_golden.graph.adj.pk',
            'adj-test2':  f'./data/{dname}/graph/11B2_golden.graph.adj.pk',
            'adj-test3':  f'./data/{dname}/graph/11B3_golden.graph.adj.pk',
            'adj-test4':  f'./data/{dname}/graph/11B4_golden.graph.adj.pk',
        },
        'tensors': {
            'tensors-train':  f'./data/{dname}/tensors/training11b.saved_tensors.pt',
            'tensors-dev':  f'./data/{dname}/tensors/dev11b_bm25.saved_tensors.pt',
            'tensors-test1':  f'./data/{dname}/tensors/11B1_golden.saved_tensors.pt',
            'tensors-test2':  f'./data/{dname}/tensors/11B2_golden.saved_tensors.pt',
            'tensors-test3':  f'./data/{dname}/tensors/11B3_golden.saved_tensors.pt',
            'tensors-test4':  f'./data/{dname}/tensors/11B4_golden.saved_tensors.pt',
        },
    }

for dname in ['mixed']:
    output_paths[dname] = {
        'statement': {
            'mini':  f'./data/{dname}/statement/mini_100.statement.jsonl',
            'train':  f'./data/{dname}/statement/mixed.statement.jsonl',
            'dev':  f'./data/{dname}/statement/mixed_dev.statement.jsonl',
        },
        'graph': {
            'adj-mini':  f'./data/{dname}/graph/mini.graph.adj.pk',
            'adj-train':  f'./data/{dname}/graph/mixed.graph.adj.pk',
            'adj-dev':  f'./data/{dname}/graph/mixed_dev.graph.adj.pk',
        },
        'tensors': {
            'tensors-mini':  f'./data/{dname}/tensors/mini.saved_tensors.pt',
            'tensors-train':  f'./data/{dname}/tensors/mixed.saved_tensors.pt',
            'tensors-dev':  f'./data/{dname}/tensors/mixed_dev.saved_tensors.pt',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['mixed'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'BioASQ': [
        #     # {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['test1'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test1'], output_paths['BioASQ']['tensors']['tensors-test1'], args.nprocs)},
        #     # {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['test2'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test2'], output_paths['BioASQ']['tensors']['tensors-test2'], args.nprocs)},
        #     # {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['test3'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test3'], output_paths['BioASQ']['tensors']['tensors-test3'], args.nprocs)},
        #     # {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['test4'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test4'], output_paths['BioASQ']['tensors']['tensors-test4'], args.nprocs)},
        #     # {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-train'], output_paths['BioASQ']['tensors']['tensors-train'], args.nprocs)},
        #     # {'func': check_bm25_entities, 'args': (output_paths['BioASQ']['statement']['test1'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test1'], output_paths['BioASQ']['tensors']['tensors-test1'], args.nprocs)},
        #     # {'func': check_bm25_entities, 'args': (output_paths['BioASQ']['statement']['test2'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test2'], output_paths['BioASQ']['tensors']['tensors-test2'], args.nprocs)},
        #     # {'func': check_bm25_entities, 'args': (output_paths['BioASQ']['statement']['test3'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test3'], output_paths['BioASQ']['tensors']['tensors-test3'], args.nprocs)},
        #     # {'func': check_bm25_entities, 'args': (output_paths['BioASQ']['statement']['test4'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-test4'], output_paths['BioASQ']['tensors']['tensors-test4'], args.nprocs)},
        #     # {'func': check_bm25_entities, 'args': (output_paths['BioASQ']['statement']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-train'], output_paths['BioASQ']['tensors']['tensors-train'], args.nprocs)},
        #     {'func': check_bm25_questions, 'args': (output_paths['BioASQ']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['BioASQ']['graph']['adj-dev'], output_paths['BioASQ']['tensors']['tensors-dev'], args.nprocs)},
        # ],
        'mixed' :[
            {'func': check_bm25_questions, 'args': (output_paths['mixed']['statement']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['mixed']['graph']['adj-dev'], output_paths['mixed']['tensors']['tensors-dev'], args.nprocs)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))

if __name__ == '__main__':
    main()