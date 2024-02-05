import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

repo_root = '/storage3/gkou/lm_graph/lm_graph'

#Prepare `statement` data following CommonsenseQA, OpenBookQA
medqa_root = f'{repo_root}/data/medqa'
os.system(f'mkdir -p {medqa_root}/statement')

for fname in ["train", "dev", "test"]:
    with open(f"{medqa_root}/raw/questions/US/4_options/phrases_no_exclude_{fname}.jsonl") as f:
        lines = f.readlines()
    examples = []
    for i in tqdm(range(len(lines))):
        line = json.loads(lines[i])
        _id  = f"train-{i:05d}"
        answerKey = line["answer_idx"]
        stem      = line["question"]    
        choices   = [{"label": k, "text": line["options"][k]} for k in "ABCD"]
        stmts     = [{"statement": stem +" "+ c["text"]} for c in choices]
        ex_obj    = {"id": _id, 
                     "question": {"stem": stem, "choices": choices}, 
                     "answerKey": answerKey, 
                     "statements": stmts
                    }
        examples.append(ex_obj)
    with open(f"{medqa_root}/statement/{fname}.statement.jsonl", 'w') as fout:
        for dic in examples:
            print (json.dumps(dic), file=fout)