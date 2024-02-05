import json
import torch
import time

def split_jsonl(inputfolder, outputfolder, file_path, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    # Read the JSONL file
    with open(inputfolder + file_path, 'r') as file:
        lines = file.readlines()
        
    # Calculate split indices
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    dev_end = train_end + int(total_lines * dev_ratio)

    # Split the data
    train_data = lines[:train_end]
    dev_data = lines[train_end:dev_end]
    test_data = lines[dev_end:]
    
    # Save the splits to new JSONL files
    with open(outputfolder + 'train.grounded.jsonl', 'w') as file:
        file.writelines(train_data)
    with open(outputfolder + 'dev.grounded.jsonl', 'w') as file:
        file.writelines(dev_data)
    with open(outputfolder + 'test.grounded.jsonl', 'w') as file:
        file.writelines(test_data)

if __name__ == "__main__":

    __spec__ = None

    # inputfolder = "/storage3/gkou/lm_graph/lm_graph/data/pubmedqa/raw/"
    # outputfolder = "/storage3/gkou/lm_graph/lm_graph/data/pubmedqa/grounded/"
    # split_jsonl(inputfolder, outputfolder, "raw_dataset.json")

    initial_memory = torch.cuda.memory_allocated()
    start = time.time()
    loaded_tensor = torch.load('data/pubmedqa/tensors/train_midle.saved_tensors.pt', map_location='cuda:2')
    print("load:", time.time()-start)
    memory_used = torch.cuda.memory_allocated() - initial_memory
    print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB") 
    print("After loading:")
    # print(loaded_tensor)
    # for id in loaded_tensor:
    #     # print(id)
    #     print(f"{id}:", loaded_tensor[id])
        


    # Now you can extract the row indices and check for the specific index 13923664
    # row_index_to_check = 13923664  # Example index you want to check
    # Checking if the specific index exists
    # index_exists = row_index_to_check in row_indices

    # print(f"Row index {row_index_to_check} exists: {index_exists}")

with open("data/pubmedqa/statement/train_midle.statement.jsonl", 'r', encoding='utf-8') as fin_ground:
    lines_ground = fin_ground.readlines()
    for j, line in enumerate(lines_ground):
        obj = json.loads(lines_ground[j])
        sparse_tensor = loaded_tensor[obj['id']]
        row_indices = sparse_tensor._indices()[0].tolist()
        print(len(row_indices))
        row_indices = [int(x) for x in row_indices]
        row_index_to_check = int(obj['ans'])
        print("row_index_to_check", row_index_to_check)
        index_exists = row_index_to_check in row_indices
        print(f"Row index {obj['id']} exists: {index_exists}")


question = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"

from elastic import elastic_search_query
results = elastic_search_query.elastic_search_text(question, 100)

for idx, res in enumerate(results['hits']['hits']):
    
    if int(res['_id']) == 11847462:
        print(res['_source']['graph_entities'])