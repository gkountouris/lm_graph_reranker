import requests
import gzip
import io
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

def main(umls_vocab_path, output_path):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")
    
    with open(output_path + "/training12b_new.json", "r") as f:

        data = json.load(f)

        counter = 0 
        questions = []
        for i in data['questions']:
            questions.append(i['body'])
            if counter > 15:
                break
        f.close()

    articles_data = []
    
    for q in questions:

        linked_entities = ground_abstract(q)
        articles_data.append({
                        'question': q,
                        'graph_entities': linked_entities
                    })
    # Store the concatenated information in a JSON file
    with open(output_path + "/BioASQ_grounded.json", "w") as f:
        json.dump(articles_data, f, indent=4)

if __name__ == "__main__":
    vocab = './data/umls/concepts.txt'
    output_path = './data/BioASQ'
    # main(vocab, output_path)
    number_of_empty(output_path)