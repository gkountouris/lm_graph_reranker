import json
import matplotlib.pyplot as plt
from collections import Counter

def read_json(path):
    with open(path, 'r') as fin:
        lines = [line for line in fin]
    relations = []
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        relations = relations + j["qc"]
    return relations

def hist_plot(relations):

    # Count the occurrences of each distinct element
    element_counts = Counter(relations)

    # Step 1: Read the txt file and create a dictionary mapping codes to names
    code_to_name = {}
    with open('./data/umls/concept_names.txt', 'r') as f:
        for line in f:
            code, name = line.strip().split('\t')
            code_to_name[code] = name

    # Step 2: Replace codes in the data dictionary with names
    named_elements = {code_to_name.get(k, k): v for k, v in element_counts.items()}

    for i in range(20):

        sorted_elements = sorted(named_elements.items(), key=lambda x: x[1], reverse=True)[1:40]

        print(sorted_elements)

        # Sort elements by count in descending order and take the top 40
        sorted_elements = sorted(named_elements.items(), key=lambda x: x[1], reverse=True)[10*i:10*(i+1)]
        elements, counts = zip(*sorted_elements)

        # Plot the histogram for top 40 elements
        plt.figure(figsize=(12, 6))
        plt.bar(elements, counts, color='blue')
        plt.xlabel('Elements')
        plt.ylabel('Count')
        plt.title('Histogram of Top 40 Element Counts')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a file
        filename = f"histogram_plot{i}.png"
        plt.savefig(filename)

if __name__ == "__main__":
    # downloadpubmed()
    relations = read_json("data/medqa/grounded/dev.grounded_abstracts.jsonl")
    hist_plot(relations)