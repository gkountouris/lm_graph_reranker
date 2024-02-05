import numpy as np

# Example data
data_dict = {
    '92013': [
        (8794, 0.05527611974690432), (7817, 0.04151992434143876), (8564, 0.039720355023678405),
        # ... more tuples ...
    ]
}

# Example vectors (assuming a list of vectors where the index corresponds to the first element of the tuple)
example_vectors = np.random.rand(100000, 10)  # Change this to your actual vectors

def weighted_similarities(data, vectors):
    for pmid, tuples in data.items():
        # Extract indices and weights from the tuples
        indices = np.array([t[0] for t in tuples])
        weights = np.array([t[1] for t in tuples])

        # Retrieve the relevant vectors and multiply by weights
        selected_vectors = vectors[indices]
        weighted_vectors = selected_vectors * weights[:, np.newaxis]

        # Sum the weighted vectors
        result_vector = np.sum(weighted_vectors, axis=0)

        yield pmid, result_vector

# Process the data
processed_data = dict(weighted_similarities(data_dict, example_vectors))

# Show results
for pmid, vec in processed_data.items():
    print(f"PMID: {pmid}, Resulting Vector: {vec}")