from elasticsearch import Elasticsearch
import os
import json
from tqdm import tqdm
import warnings
from elasticsearch import ElasticsearchWarning

# Suppress Elasticsearch warnings
warnings.filterwarnings('ignore', category=ElasticsearchWarning)


def elastic_search_text(query, number_of_results):

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

    query_body = {
        "size": number_of_results,
        "query": {
            "match": {
                "text": query
            }
        }
    }
    
    return es.search(index=index_name, body=query_body)


def elastic_search_PMID(pmid):

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

    query_body = {
        "size": 1,
        "query": {
            "match": {
                "PMID": pmid
            }
        }
    }
    
    return es.search(index=index_name, body=query_body)

def elastic_search_ID(id):

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

    query_body = {
        "size": 1,
        "query": {
            "match": {
                "_id": id
            }
        }
    }
    
    return es.search(index=index_name, body=query_body)


def elastic_search_entities(graph_entities):

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

    # Elasticsearch query
    query_body = {
        "query": {
            "terms": {
            "graph_entities": graph_entities
            }
        }
    }

    return  es.search(index=index_name, body=query_body)

if __name__ == "__main__":

    vocab = './data/umls/concepts.txt'
    pubmed_path = './data/pubmed_processed'
    index_name = 'pubmed_documents'
    # query = 'Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through?'
    # query = 'Is the protein Papilin secreted?'
    # query = 'Are long non coding RNAs spliced?'
    query = 'Which receptor is targeted by Erenumab?'

    results = elastic_search_text(query, 100)

    pmid_list = []
    total_entities = set()
    true_entities = set()
    total = 0
    for res in results['hits']['hits']:
        pmid_list.append(res['_source']['PMID'])
        total += len(res['_source']['graph_entities'])
        for ent in res['_source']['graph_entities']:
            total_entities.add(ent)
            # if res['_source']['PMID'] in true_pmids:
            #     true_entities.add(ent)

    # print(pmid_list)
    print(total)
    print(len(total_entities))

    # true_pmids = ['3320045', '7515725', '20805556', '19297413', '19724244', '15094122', '12666201', '21784067', '11076767', '15094110']
    true_pmids = ['22955988', '22955974', '24285305', '22707570', '21622663', '24106460', '12666201', '21784067', '11076767', '15094110']


    results = elastic_search_PMID("21")
    print(results)
    # for pmid in true_pmids:
    #     true_entities = set()
    #     results = elastic_search_PMID(es, index_name, pmid)
    #     for res in results['hits']['hits']:
    #         print(res['_source']['text'])
    #         print(res['_source']['graph_entities'])
    #         for ent in res['_source']['graph_entities']:
    #             true_entities.add(ent)
    #     print(len(true_entities.difference(total_entities)), len(true_entities))
    # for res in results['hits']['hits']:
    #     true_entities = set()
    #     if res['_source']['PMID'] in true_pmids:
    #         for ent in res['_source']['graph_entities']:
    #                 true_entities.add(ent)
    #         print(true_entities)
    #         print(len(true_entities.difference(total_entities)))

    
    # print(total_entities)
