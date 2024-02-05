import requests
import gzip
import io
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import json
from elastic import elastic_search_query
from multiprocessing import Pool
import spacy
# from spacy.matcher import Matcher
from tqdm import tqdm
import os
import nltk
import json
import string

import scispacy

import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict

UMLS_VOCAB = None
nlp = None
linker = None

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

def load_umls_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as fin:
        vocab = [l.strip() for l in fin]
    return vocab

def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    linker = EntityLinker(
        resolve_abbreviations=True,
        name="umls",
        threshold=threshold)
    nlp.add_pipe(linker)
    return nlp, linker

def entity_linking_to_umls(sentence, nlp, linker):
    try:
        doc = nlp(sentence)
        entities = doc.ents
    except:
        entities = []
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
        
    return all_entities_results

def ground_mentioned_concepts(nlp, linker, sent):
    ent_link_results = entity_linking_to_umls(sent, nlp, linker)
    mentioned_concepts = set()
    for ent_obj in ent_link_results:
        for ent_cand in ent_obj['linking_results']:
            CUI = ent_cand['Concept ID']
            if CUI in UMLS_VOCAB:
                mentioned_concepts.add(CUI)
    return mentioned_concepts

def ground_abstract(line):
    
    question_concepts = ground_mentioned_concepts(nlp, linker, line)
    question_concepts = sorted(list(question_concepts))

    return question_concepts

def number_of_empty(output_path):

    # Load your JSON data
    with open(output_path + '/BioASQ_grounded.json', 'r') as file:
        data = json.load(file)

    # Iterate over the data and check for empty graph_entities
    empty_graph_entities = []
    with_graph_entities = []
    for idx, item in enumerate(data):
        if not item['graph_entities']:  # Check if graph_entities is empty
            empty_graph_entities.append(idx)
        else:
            with_graph_entities.append(idx)

    print("Documents with empty graph_entities:", len(with_graph_entities))
    print("Documents with empty graph_entities:", len(empty_graph_entities))

def transform_json(output_path, process_json):

    with open(output_path + '/grounded' + process_json, "r") as f:
        data = json.load(f)

    for item in data:
        print(item)

    # Assuming articles_data is a list of dictionaries
    with open(output_path + '/grounded/' + 'train.grounded.jsonl', "w") as f:
        for item in data:
            json.dump({'sent': item['question'], 'qc': item['graph_entities'], 'ans' : item['PMID']}, f)
            f.write('\n')  # Write a newline character after each JSON object

def main(umls_vocab_path, output_path, process_json1, process_json2, process_json3, process_output):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")

    with open(output_path + process_json1, "r") as f:

        data1 = json.load(f)

    with open(output_path + process_json2, "r") as f:

        data2 = json.load(f)

    with open(output_path + process_json3, "r") as f:

        data3 = json.load(f)
        
    articles_data = []
    for data in [data1, data2, data3]:
        for key, _ in tqdm(data.items()):
            try:
                question = data[key]["QUESTION"]
                results = elastic_search_query.elastic_search_PMID(key)
                id = results['hits']['hits'][0]['_id']
                linked_entities = ground_abstract(question)
                articles_data.append({
                                    'sent': question,
                                    'qc': linked_entities,
                                    'ans' : id
                                })
            except:
                pass
    
    # Assuming articles_data is a list of dictionaries
    with open(output_path  + process_output, "w") as f:
        for item in articles_data:
            json.dump(item, f)
            f.write('\n')  # Write a newline character after each JSON object


if __name__ == "__main__":

    vocab = './data/umls/concepts.txt'
    output_path = './data/pubmedqa'

    process_json1 = '/raw/ori_pqal.json'
    process_json2 = '/raw/ori_pqau.json'
    process_json3 = '/raw/ori_pqaa.json'
    process_output = '/raw/raw_dataset.json'
    # main(vocab, output_path, process_json1, process_json2, process_json3, process_output)

    inputfolder = "/storage3/gkou/lm_graph/lm_graph/data/pubmedqa/raw/"
    outputfolder = "/storage3/gkou/lm_graph/lm_graph/data/pubmedqa/grounded/"
    split_jsonl(inputfolder, outputfolder, "raw_dataset.json")
    
    # transform_json(output_path, process_json)
    # process_json = '/ori_pqaa.json'
    # main(vocab, output_path, process_json)
    # process_json = '/ori_pqau.json'
    # main(vocab, output_path, process_json)


