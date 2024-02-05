import torch
import time
from scipy.sparse import load_npz
import numpy as np

def process_tf_idf_part(tf_idf_part, sublist):

    def slice_sparse_tensor(sparse_tensor, col_indices):
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()

        # Create a mask for the columns to select
        mask = torch.isin(indices[1], col_indices)

        # Apply the mask to select the corresponding values and indices
        selected_indices = indices[:, mask]
        selected_values = values[mask]

        # Ensure col_indices is sorted
        unique_col_indices, _ = torch.sort(col_indices)

        # Use torch.searchsorted to find positions
        remapped_cols = torch.searchsorted(unique_col_indices, selected_indices[1])

        # Create the new indices tensor with remapped column indices
        new_indices = torch.stack([selected_indices[0], remapped_cols])

        # The shape should have the same number of rows and the length of col_indices
        shape = [sparse_tensor.shape[0], len(col_indices)]

        # Create the new sparse tensor
        selected_tensor = torch.sparse_coo_tensor(new_indices, selected_values, torch.Size(shape)) #, requires_grad=True

        return selected_tensor
    
    # Manually slice the sparse tensor
    selected_cols_matrix = slice_sparse_tensor(tf_idf_part, sublist)
    
    return selected_cols_matrix

# # Simulate the sizes of your tensors
# num_rows = 23662581
# num_cols = 200
dense_size = 200

# # Number of non-zero elements (nnz)
# nnz = 299139

# # Create random indices within the bounds of the tensor dimensions
# row_indices = torch.randint(0, num_rows, (nnz,))
# col_indices = torch.randint(0, num_cols, (nnz,))

# # Stack the indices to get a 2D tensor
# indices = torch.stack([row_indices, col_indices], dim=0)

# # Create random values for the sparse tensor
# values = torch.randn(nnz)

# # Create the sparse tensor
# sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([num_rows, num_cols])).cuda().requires_grad_(True)

# # Create a dense tensor with random values
dense_tensor = torch.randn(dense_size, 1).cuda().requires_grad_(True)

dense_tensor = dense_tensor.to(torch.device("cpu"))

# # Measure initial memory usage
# initial_memory = torch.cuda.memory_allocated()
# result = torch.sparse.mm(sparse_tensor, dense_tensor)
# # Measure memory usage after the operation
# memory_used = torch.cuda.memory_allocated() - initial_memory

# # Perform sparse matrix multiplication

# print(dense_tensor)
# print(sparse_tensor)
# print(f"Initial Memory used: {initial_memory / (1024 ** 2):.2f} MB")  # Convert to MB for readability
# print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")  # Convert to MB for readability

# print(result)

import scipy.sparse as sp

# Values for the tensor
values = [0, 110564, 53136, 198364, 270921, 216403, 228489, 262617, 177420,
          133703, 269487, 207231, 53134, 234142, 234167, 268839, 30916, 266337,
          183209, 261181, 53135, 245464, 254483, 181984, 270366, 283354, 283356,
          269683, 216844, 270373, 30913, 53144, 265774, 192454, 30885, 30914,
          216928, 270372, 210314, 198365, 264255, 195639, 259549, 270365, 270364,
          210877, 216784, 254924, 270369, 270374, 259446, 288550, 231644, 263979,
          21456, 283487, 150925, 229625, 172300, 229213, 270408, 297328, 177556,
          270900, 259456, 270368, 241808, 95264, 228263, 273359, 266269, 140495,
          140262, 297608, 201477, 179719, 128641, 30859, 47975, 271228, 288065,
          98775, 98776, 259393, 259370, 95266, 53163, 270920, 266341, 273399,
          175494, 273401, 192759, 207474, 218846, 21344, 266679, 285362, 233912,
          234233, 236197, 53142, 30895, 45354, 244316, 102421, 43434, 236043,
          83979, 265492, 177456, 218982, 258903, 231899, 242267, 53133, 53140,
          222097, 235234, 270583, 226272, 226894, 211024, 260968, 65528, 3210,
          34171, 218979, 261028, 270938, 34661, 288465, 266672, 192453, 263032,
          259782, 264262, 173402, 235486, 219612, 261011, 263985, 92805, 228582,
          142819, 247436, 229148, 266396, 132553, 202860, 297581, 259464, 207487,
          180224, 244288, 270899, 21351, 236399, 285368, 34327, 93474, 53148,
          95312, 234172, 285388, 265505, 270397, 236247, 242266, 258845, 226940,
          229947, 167998, 53143, 229883, 256103, 280778, 31071, 217309, 252227,
          86131, 106506, 270406, 34522, 288455, 92969, 214995, 272011, 270405,
          233771, 256606, 280777, 265198, 266404, 260983, 235876, 195694, 214024,
          258954, 31325]

# Create a tensor from the list of values
sublist = torch.tensor(values, device='cuda:0')

# Load Tf_Idf matrix
tf_idf_matrix = load_npz("data/pubmed/sparse_matrix.npz")

tf_idf_matrix = tf_idf_matrix.tocsr()

num_devices = 1

# Split the matrix into parts equal to the number of devices
split_indices = np.linspace(0, tf_idf_matrix.shape[0], num_devices + 1, dtype=int)

initial_memory = torch.cuda.memory_allocated()
torch_tf_idf_parts = []

for idx in range(num_devices):
    start, end = split_indices[idx], split_indices[idx + 1]
    part = tf_idf_matrix[start:end, :].tocoo()

    # Adjust row indices to reflect original position
    adjusted_row_indices = part.row + start

    # Create PyTorch sparse tensor with adjusted indices
    i = torch.LongTensor(np.vstack((adjusted_row_indices, part.col)))
    v = torch.FloatTensor(part.data)
    shape = (tf_idf_matrix.shape[0], part.shape[1])  # Keep original number of rows
    torch_part = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(devices[device_to_use[idx]])
    # torch_part = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.device("cpu"))
    torch_tf_idf_parts.append(torch_part)


start_time = time.time()

for torch_part in torch_tf_idf_parts:
    sublist = sublist.to(torch_part.device) 
    selected_cols_matrix = process_tf_idf_part(torch_part, sublist)

result = torch.sparse.mm(selected_cols_matrix, dense_tensor)
memory_used = torch.cuda.memory_allocated() - initial_memory

end_time = time.time()
print("multi: ", end_time - start_time)

torch.save(selected_cols_matrix, 'selected_cols_matrix.pt')

print(f"Initial Memory used: {initial_memory / (1024 ** 2):.2f} MB")  # Convert to MB for readability
print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")  # Convert to MB for readability

dense_tensor = dense_tensor.cuda()

initial_memory = torch.cuda.memory_allocated()
start_time = time.time()
selected_cols_matrix = torch.load('selected_cols_matrix.pt', map_location='cuda:0')
result = torch.sparse.mm(selected_cols_matrix, dense_tensor)
end_time = time.time()
print("Saved multi: ", end_time - start_time)
memory_used = torch.cuda.memory_allocated() - initial_memory
print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")  # Convert to MB for readability