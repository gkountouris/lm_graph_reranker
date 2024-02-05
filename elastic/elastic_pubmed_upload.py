from elasticsearch import Elasticsearch
import os
import json
from elasticsearch.helpers import bulk
from tqdm import tqdm


def es_connect():
    ip="http://localhost:9200"
    es = Elasticsearch([
            ip
            ],
                verify_certs=True,
                timeout=150,
                max_retries=10,
                retry_on_timeout=True
            )

    if es.ping():
        print("Connected")
    else:
        print("Unable to connect")

    return es


def create_index(es, index_name):

    es.indices.delete(index=index_name)

    mapping = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 2
        },
        "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "PMID": {"type": "keyword"},
                    "mesh_headings": {"type": "text"},
                    "mesh_headings_list": {"type": "text"},
                    "keywords": {"type": "text"},
                    "keywords_list": {"type": "text"},
                    "pub_date": {"type": "date", "format": "year", "null_value": "NULL"},
                    "iso_abbreviation": {"type": "text"},
                    "substances": {"type": "text"},
                    "graph_entities": {"type": "keyword"},
                    "entities": {"type": "text"}
                }
            }
        }

    es.indices.create(index=index_name, body=mapping)


def upload_pubmed_elastic_file(es, pubmed_path, index_name):

    vocab = 'data/umls/concept_names.txt'

    context_dict = {}
    with open(vocab, 'r') as f:
        for line in f:
            clean_line = line.strip().split("\t")
            if len(clean_line) == 2:  # Ensure there are two elements in the line
                context_dict[clean_line[0]] = clean_line[1]

    # Process JSON files
    json_files = [f for f in os.listdir(pubmed_path) if f.endswith('.json')]

    index = 0
    total_files = 1167

    for i in tqdm(range(1, total_files)):
        with open(pubmed_path + f"/pubmed23n{i:04}.json", "r") as json_file:
            json_docs = json.load(json_file)

    # # we need both the json and an index number so use enumerate()
    # for js in tqdm(json_files):
    #     with open(os.path.join(pubmed_path, js)) as json_file:
    #         json_docs = json.load(json_file)

        # Prepare documents for bulk insertion
        actions = []
        for doc in json_docs:
            index += 1
            doc['mesh_headings_list'] = doc['mesh_headings'].split(", ")
            doc['keywords_list'] = doc['keywords'].split(", ")
            if not doc['pub_date'].strip():
                doc['pub_date'] = None
            name_entities = []
            for entities in doc['graph_entities']:
                name_entities.append(context_dict[entities])
            doc['entities'] = name_entities

            action = {
                "_index": index_name,
                "_id": index,
                "_source": doc
            }
            actions.append(action)
            
            # Bulk insert in batches of 1000
            if len(actions) >= 1000:
                bulk(es, actions)
                actions = []

        # Insert any remaining documents
        if actions:
            bulk(es, actions)


if __name__ == "__main__":
    vocab = './data/umls/concepts.txt'
    pubmed_path = './data/pubmed_processed'
    index_name = 'pubmed_documents'

    es = es_connect()
    create_index(es, index_name)
    upload_pubmed_elastic_file(es, pubmed_path, index_name)