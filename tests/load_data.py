import numpy as np
import os
import pickle
import networkx as nx
import json

# def read_first_n_lines(filename, n=3):
#     lines = []
#     with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
#         for _ in range(n):
#             line = file.readline().strip()
#             if not line:
#                 break
#             lines.append(line)
#     return lines

# folder_path = "/storage3/gkou/lm_graph/lm_graph/data/umls"
# files = ["concept_names.txt", "concepts.txt", "ent_emb_blbertL.npy", "relations.txt", "umls.csv", "umls.graph"]

# n = 3

# with open('/storage3/gkou/lm_graph/lm_graph/data/umls/umls.graph', 'rb') as file:
#     graph = pickle.load(file)

# # Now you can inspect the graph
# print(type(graph))
# print(graph.number_of_nodes())
# print(graph.number_of_edges())

# for node, data in list(graph.nodes(data=True))[:3]:
#     print("nodes")
#     print(node, data)
#     print("-----")
# for source, target, data in list(graph.edges(data=True))[:3]:
#     print("source target")
#     print(source, target, data)
#     print("-----")


# for filename in files:
#     file_path = os.path.join(folder_path, filename)
#     print(f"First {n} lines of {filename}:")
    
#     if filename.endswith(".npy"):
#         array = np.load(file_path)
#         for i in range(3):
#             if i < len(array):
#                 print(array[i])
#     else:
#         try:
#             lines = read_first_n_lines(file_path, n)
#             for line in lines:
#                 print(line)
#         except UnicodeDecodeError:
#             print(f"{filename} might not be encoded in UTF-8 or could be a binary file.")
    
#     print("-----")


file_path = '/storage3/gkou/lm_graph/lm_graph/data/BioASQ/allMeSH_2022.json'  # replace with the path to your JSON file


with open(file_path, 'r', encoding='latin-1') as f:
     data = json.load(f)

for keys in data:
     print(keys)
