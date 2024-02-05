import json
from elastic import elastic_search_query

# Path to your JSON file
path = "/storage3/gkou/lm_graph/lm_graph/data/pubmedqa/statement/dev.statement.jsonl"

# Open the input file and read line by line
with open(path, 'r') as file:
    lines = file.readlines()

# Process each line
processed_lines = []
for i, line in enumerate(lines):
    # Parse the line as JSON
    json_obj = json.loads(line)

    json_obj['id'] = f"train-{i:06}"  # Assuming you want to use a simple incremental ID

    # Convert back to JSON string
    processed_line = json.dumps(json_obj)

    # Add the processed line to the list
    processed_lines.append(processed_line)



# Write the processed lines to the output file
with open(path, 'w') as file:
    for line in processed_lines:
        file.write(line + '\n')

