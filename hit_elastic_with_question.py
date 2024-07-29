
from elastic import elastic_search_query
import torch


question = "What is CHARMS with respect to medical review of predictive modeling?"

results = elastic_search_query.elastic_search_text(question, graph_entities=None, id_list=10)

bm_labels = []
for idx, res in enumerate(results['hits']['hits']):
    bm_labels.append(int(res['_source']['PMID']))

print(bm_labels)

