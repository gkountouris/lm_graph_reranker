from rank_bm25 import BM25Okapi
import json
from multiprocessing import Pool
from tqdm import tqdm

def check_file_for_PMID(args):
    file_path, pmid = args
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            for dato in data:
                if dato["PMID"] == pmid:
                    return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return False

def find_PMID(pmid, num_processes=4):
    output_path = './data/pubmed_processed'
    file_paths = [output_path + f"/pubmed23n{i:04}.json" for i in range(1165, 0, -1)]
    
    with Pool(num_processes) as p:
        results = list(tqdm(p.imap(check_file_for_PMID, [(path, pmid) for path in file_paths]), total=len(file_paths)))

    for i, result in enumerate(results):
        if result:
            return 1165 - i  # Adjust index based on your file numbering
    return None
        
def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return [(doc['PMID'], doc['text'].split(" ")) for doc in data]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def BM25_func(query, num_processes=4):
    output_path = './data/pubmed_processed'
    file_paths = [output_path + f"/pubmed23n{i:04}.json" for i in range(1166, 0, -1)]
    
    with Pool(num_processes) as p:
        results = list(tqdm(p.imap(process_file, file_paths), total=len(file_paths)))

    # Flatten results and separate PMIDs and tokenized texts
    abstracts_with_pmids = [item for sublist in results for item in sublist]
    tokenized_abstracts = [doc[1] for doc in abstracts_with_pmids]
    pmids = [doc[0] for doc in abstracts_with_pmids]

    # Create BM25 object and get scores
    bm25 = BM25Okapi(tokenized_abstracts)
    scores = bm25.get_scores(query.split(" "))  # Tokenize query

    # Get top N relevant abstracts
    N = 20
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    top_n_pmids = [pmids[i] for i in top_n_indices]

    return top_n_pmids


if __name__ == "__main__":
    __spec__ = None

    # Example usage
    pmid_to_find = '11178982' 
    file_index = find_PMID(pmid_to_find)
    if file_index is not None:
        print(f"PMID {pmid_to_find} found in file number: {file_index}")
    else:
        print(f"PMID {pmid_to_find} not found in any file.")

    # Query
    # query = "Is vemurafenib effective for hairy-cell leukemia?"
    # top_n_pmids = BM25_func(query)
    # print(top_n_pmids)


