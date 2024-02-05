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



pubmedpath = '/storage3/gkou/lm_graph/lm_graph/data/pubmed_processed'

foundit = False
i = 1129
while not foundit:
    i -= 1
    # Store the concatenated information in a JSON file
    with open(pubmedpath + f"/pubmed23n{i:04}.json", "r") as f:
        data = json.load(f)

        for dato in data: 
            # if dato['PMID'] == '26191653':
            if dato['PMID'] == '26259533':
                print(dato)
                foundit = True
                break