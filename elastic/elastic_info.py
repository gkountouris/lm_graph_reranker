from elasticsearch import Elasticsearch

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

# Elasticsearch info
print('##########################################')
print('###################info###################')
print('##########################################')
print(es.info())

# Get the statistics for all indices
indices_stats = es.indices.stats()

for indexes in indices_stats['indices']:
    print('For the index: ', indexes)
    for index_keys in indices_stats['indices'][indexes]:
        if index_keys == 'health':
            print('############## health: ', indices_stats['indices'][indexes][index_keys])
        if index_keys == 'status':
            print('############## status: ', indices_stats['indices'][indexes][index_keys])
        if index_keys == 'primaries':
            print('############## primaries: ')
            for total_keys in indices_stats['indices'][indexes][index_keys]:
                if total_keys == 'docs':
                    print('############## ############## Docs count: ', indices_stats['indices'][indexes][index_keys][total_keys]['count'])
                if total_keys == 'shard_stats':
                    print('############## ############## Shard count: ', indices_stats['indices'][indexes][index_keys][total_keys]['total_count'])
                if total_keys == 'store':
                    print('############## ############## Size_in_bytes: ', indices_stats['indices'][indexes][index_keys][total_keys]['size_in_bytes'])
        if index_keys == 'total':
            print('############## total: ')
            for total_keys in indices_stats['indices'][indexes][index_keys]:
                if total_keys == 'docs':
                    print('############## ############## Docs count: ', indices_stats['indices'][indexes][index_keys][total_keys]['count'])
                if total_keys == 'shard_stats':
                    print('############## ############## Shard count: ', indices_stats['indices'][indexes][index_keys][total_keys]['total_count'])
                if total_keys == 'store':
                    print('############## ############## Size_in_bytes: ', indices_stats['indices'][indexes][index_keys][total_keys]['size_in_bytes'])

# Get the health of the cluster
print('##########################################')
print('#################Health###################')
print('##########################################')
print(es.cluster.health())

# Check the indexing status
if indices_stats['_all']['primaries']['indexing']['index_total'] == 0:
    print('No new documents are being indexed.')
else:
    print(f"{indices_stats['_all']['primaries']['indexing']['index_total']} documents have been indexed.")

# Search for logs
print('##########################################')
print('##################LOGS####################')
print('##########################################')
logs = es.search(index='logstash-*', body={
    'query': {
        'match': {'message': 'error'}
    }
})

# Print the logs
for hit in logs['hits']['hits']:
    print(hit['_source']['message'])