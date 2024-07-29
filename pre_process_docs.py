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
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")
    
    question_concepts = ground_mentioned_concepts(nlp, linker, line)
    if len(question_concepts) == 0:
        print(f"Concept not found in umls linking.")

    question_concepts = sorted(list(question_concepts))

    return {"sent": line, "qc": question_concepts}

def read_pubmed_json():
    with open("articles_data.json", "r") as f:
        data = json.load(f)
        # Assuming you want to print the first 3 articles as separate lines
        for article in data[:3]:
            print(json.dumps(article, indent=4))

def ground_umls(statement_path, umls_vocab_path, output_path, num_processes=1, debug=False):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))

    with open(statement_path, 'r') as fin:
        data = json.load(fin)

    res = []
    for dato in tqdm(data):
        txt_string = dato['text']
        for subs in dato['substances']:
            txt_string = txt_string + ' ' + subs + '.'
        if txt_string == "":
            continue
        res.append(ground_abstract(txt_string))
    print(res['qc'])
    # os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    # with open(output_path, 'w') as fout:
    #     for dic in res:
    #         fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    statement = './articles_data.json'
    vocab = './data/umls/concepts.txt'
    grounded = './data/medqa/grounded/dev.grounded_abstracts.jsonl'
    ground_umls(statement, vocab, grounded, debug=True)




# def ground_umls(statement_path, umls_vocab_path, output_path, num_processes=1, debug=False):
#     global UMLS_VOCAB
#     if UMLS_VOCAB is None:
#         UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))

#     with open(statement_path, 'r') as fin:
#         data = json.load(fin)
    
#     res = []
#     for line in tqdm(data):
#         print(line)
#         if line == "":
#             continue
#         res.append(ground_abstract(line))
#         break

#     # os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
#     # with open(output_path, 'w') as fout:
#     #     for dic in res:
#     #         fout.write(json.dumps(dic) + '\n')

#     print(f'grounded concepts saved to {output_path}')
#     print()