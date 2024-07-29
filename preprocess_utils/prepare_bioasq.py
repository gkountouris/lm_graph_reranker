import requests
import gzip
import io
import glob
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import json

from multiprocessing import Pool
import spacy
# from spacy.matcher import Matcher
from tqdm import tqdm
import os
import nltk
import json
import string

import scispacy
from scispacy.linking import EntityLinker

UMLS_VOCAB = None
nlp = None
linker = None

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

    doc = nlp(sentence)
    entities = doc.ents
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

def rename_file(file_path):
    
    # Split the path into directory and file name
    dir_name, file_name = os.path.split(file_path)
    
    # Split the filename into name and extension
    base_name, _ = os.path.splitext(file_name)
    
    # Create the new filename by adding '.statement' before the extension
    new_file_name = base_name + '.statement.jsonl'
    
    return new_file_name

def main(umls_vocab_path, output_path):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")
    dict_path = "data/pubmed_processed/hdf5/id_pmid_mapping.json"
    with open(dict_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    pid_to_id = {v: k for k, v in docs.items()}
    
    # Create a pattern to match all .json files
    pattern = os.path.join(output_path + "/raw", '*.json')
    # Use glob.glob() to find all files that match the pattern
    json_files = glob.glob(pattern)
    for file in json_files:
        articles_data = []
        with open(file, "r") as f:
            data = json.load(f)
            for i, dato in enumerate(data['questions']):
                t_id = f"train-{i:06d}"
                question = dato['body']
                ids = []
                for url in dato['documents']:
                    try:
                        ids.append(pid_to_id[url.split('/')[-1]])
                    except:
                        pass
                answers = ids
                if not answers:
                    continue
                try:
                    linked_entities = ground_abstract(question)
                except:
                    linked_entities = []
                articles_data.append({
                                'sent': question,
                                'qc': linked_entities,
                                'ans': answers,
                                'id': t_id
                            })
            f.close()
        # Write each item in articles_data to the .jsonl file, one object per line
        with open(output_path + "/statement/" + rename_file(file), "w") as f:
            for article in articles_data:
                f.write(json.dumps(article) + '\n')  # Write each JSON object as a string followed by a newline

if __name__ == "__main__":
    vocab = './data/umls/concepts.txt'
    output_path = './data/BioASQ'
    main(vocab, output_path)