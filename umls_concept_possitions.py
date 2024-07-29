import torch
import time
from scipy.sparse import load_npz
import numpy as np

from elastic import elastic_search_query

import scipy.sparse as sp
import pickle
from tqdm import tqdm
import statistics


elastic_search_query

import json

def create_position_content_dict(file_path):
    # Initialize an empty dictionary
    position_content_dict = {}

    # Open the text file in read mode
    with open(file_path, 'r') as file:
        # Read lines one by one
        for position, line in enumerate(file, start=1):
            # Strip newline characters from the end of each line
            content = line.strip()
            # Assign the line number (position) as the key and line content as the value
            position_content_dict[position] = content

    return position_content_dict

def save_dict_to_json(data, json_file_path):
    # Save the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
file_path = 'data/umls/concepts.txt'
json_file_path = 'data/umls/concepts_positions.json'

# Create the dictionary
position_content_dict = create_position_content_dict(file_path)

# Save the dictionary as a JSON file
save_dict_to_json(position_content_dict, json_file_path)