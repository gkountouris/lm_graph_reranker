import numpy as np
from scipy.sparse import coo_matrix
import torch
from tqdm import tqdm
import time
import socket, os, sys, subprocess
from utils import parser_utils

def get_devices(args):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""

    if args.local_rank == -1 or not args.cuda:
        if torch.cuda.device_count() >= 3 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            device2 = torch.device("cuda:2")  # Add third device
            print("device0: {}, device1: {}, device2: {}".format(device0, device1, device2))
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            print("device0: {}, device1: {}".format(device0, device1))
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device0 = torch.device("cuda", args.local_rank)
        device1 = device0
        torch.distributed.init_process_group(backend="nccl")

    args.world_size = world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    print ("Process rank: %s, device: %s, distributed training: %s, world_size: %s" %
              (args.local_rank,
              device0,
              bool(args.local_rank != -1),
              world_size), file=sys.stderr)

    return device0, device1, device2 if 'device2' in locals() else device0

if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")

    args, _ = parser.parse_known_args()

    device = get_devices(args)

    print(device[1])

    # Example dimensions
    Z = 21000  # Total number of documents 
    E = 100000    # Total number of entities 

    # Create an example sparse matrix W (30 x Z)
    data = np.random.rand(30 * Z)  # Example data
    row_indices = np.random.randint(0, 30, 30 * Z)  # Example row indices
    col_indices = np.random.randint(0, Z, 30 * Z)  # Example column indices

    W_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(E, Z))

    # Convert to PyTorch sparse tensor
    indices = np.vstack((W_sparse.row, W_sparse.col))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(W_sparse.data)
    W_torch = torch.sparse_coo_tensor(indices, values, W_sparse.shape)

    # Move to GPU if available and supported
    if torch.cuda.is_available():
        W_torch = W_torch.to(device[1])
    
    # Define S matrix and split into batches
    S = np.random.randn(1, E)  # Example S matrix of shape (E, 1)
    S = torch.from_numpy(S).to_sparse()
    S = S.to(torch.float32).to(device[1])
    batch_size = int(Z)
    num_batches = Z // batch_size

    # Time measurement start
    s_time = time.time()

    # Multiply in batches
    results = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, Z)
        batch_indices = np.arange(start_idx, end_idx)

        # Extract the corresponding columns from W_torch
        # batch_W_torch = W_torch[:, batch_indices]

        # Perform multiplication
        result_batch = torch.sparse.mm(S, W_torch)
        results.append(result_batch.cpu().to_dense())

    # Concatenate results
    final_result = np.concatenate(results, axis=1)

    # Time measurement end
    e_time = time.time()
    print("Seconds =", e_time - s_time)

