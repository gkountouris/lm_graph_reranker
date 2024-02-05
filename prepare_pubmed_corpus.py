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


def main(umls_vocab_path, output_path):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")

    total_files = 1167  # Total number of files to process

    for i in range(1165, total_files):
        url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n{i:04}.xml.gz"
        print(f"Processing: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error if the download fails

        # Decompress the gzipped content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            content = gz.read().decode('utf-8')  # Decode the bytes to string

        # Parse the XML content
        root = ET.fromstring(content)

        articles_data = []

        articles = root.findall('.//PubmedArticle')
        # Iterate through articles
        for article in tqdm(articles, desc="Articles", leave=False):
            try:
                # Check if the article has an Abstract and PMID
                if article.find('.//Abstract') is not None and article.find(".//PMID") is not None:
                    title_element = article.find(".//ArticleTitle")
                    abstract_element = article.find(".//Abstract/AbstractText")
                    pmid_element = article.find(".//PMID")
                    mesh_headings = article.findall(".//MeshHeading/DescriptorName")
                    keywords = article.findall(".//KeywordList/Keyword")
                    pub_date = article.find(".//PubDate/Year")
                    iso_abbreviation = article.find(".//Journal/ISOAbbreviation")
                    substances = [substance.text for substance in article.findall(".//ChemicalList/Chemical/NameOfSubstance")]

                    # Extract the text from each element
                    title_text = title_element.text if title_element is not None else ""
                    abstract_text = abstract_element.text if abstract_element is not None else ""
                    pmid_text = pmid_element.text if pmid_element is not None else ""
                    mesh_headings_text = ", ".join([mh.text for mh in mesh_headings])
                    keywords_text = ", ".join([kw.text for kw in keywords])
                    pub_date_text = pub_date.text if pub_date is not None else ""
                    iso_abbreviation_text = iso_abbreviation.text if iso_abbreviation is not None else ""
                    
                    # Concatenate title and abstract and include PMID
                    combined_text = title_text + " " + abstract_text
                    entities_txt_string = combined_text
                    for subs in substances:
                        entities_txt_string = entities_txt_string + ' ' + subs + '.'
                    try:
                        linked_entities = ground_abstract(entities_txt_string)
                    except:
                        linked_entities = []
                    
                    articles_data.append({
                        'text': combined_text,
                        'PMID': pmid_text,
                        'mesh_headings': mesh_headings_text,
                        'keywords': keywords_text,
                        'pub_date': pub_date_text,
                        'iso_abbreviation': iso_abbreviation_text,
                        'substances': substances,
                        'graph_entities': linked_entities
                    })
            except Exception as e:
                print(f"Error : {str(e)}")
        # Store the concatenated information in a JSON file
        with open(output_path + f"/pubmed23n{i:04}.json", "w") as f:
            json.dump(articles_data, f, indent=4)


if __name__ == "__main__":
    vocab = './data/umls/concepts.txt'
    output_path = './data/pubmed_processed'
    main(vocab, output_path)