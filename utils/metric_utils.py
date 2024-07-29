import torch

def calculate_ap(bm_labels, ans, k):

    if not ans:  # If there are no relevant documents, return 0.0 to avoid division by zero
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, label in enumerate(bm_labels[:k]):
        if label in ans:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    average_precision = sum_precisions / len(ans)
    return average_precision

def calculate_mrr_at_k(bm_labels, ans, k):
    for i, label in enumerate(bm_labels[:k]):
        if label in ans:
            return 1.0 / (i + 1)  # Return the reciprocal of the rank (1-based index)
    return 0.0  # Return 0 if no relevant documents are found within the top k

def calculate_ap_tensor(bm_labels, ans, k):
    """
    Calculate the Average Precision (AP) at K for batches.
    
    Parameters:
    - bm_labels: A 2D tensor of shape (batch_size, num_labels), where each row contains labels for a batch item.
    - ans: A tensor of shape (batch_size, num_labels) or (batch_size,) with binary indicators (0 or 1) whether the label is an answer. If 1D, will be treated as (batch_size, 1).
    - k: An integer indicating the top K items to consider.
    
    Returns:
    - A tensor of shape (batch_size,) containing the AP at K for each batch item.
    """

    ans = torch.stack(ans)

    # Adjust ans if it's 1D to make it 2D
    if ans.dim() == 1:
        ans = ans.unsqueeze(-1)  # Convert to 2D if it's 1D

    k = min(k, bm_labels.size(1))
    bm_labels = bm_labels[:, :k]
    ans = ans[:, :k]

    # Generate a tensor for calculating precision at each K
    correct = ans * (bm_labels == ans).float()
    cum_correct = torch.cumsum(correct, dim=1)
    precision_at_k = cum_correct / torch.arange(1, k + 1).type_as(bm_labels).view(1, -1).expand_as(cum_correct)

    # Calculate average precision
    ap = (precision_at_k * correct).sum(dim=1) / ans.sum(dim=1).clamp(min=1)

    # Assuming calculate_mean_metric_tensor is implemented elsewhere and returns the mean AP for the batch
    return calculate_mean_metric_tensor(ap)
    

def calculate_mrr_at_k_tensor(bm_labels, ans, k):
    """
    Calculate the Mean Reciprocal Rank (MRR) at K for batches.
    
    Parameters:
    - bm_labels: A 2D tensor of shape (batch_size, num_labels), where each row contains labels for a batch item.
    - ans: A tensor of shape (batch_size, num_labels) or (batch_size,) with binary indicators (0 or 1) whether the label is an answer. If 1D, will be treated as (batch_size, 1).
    - k: An integer indicating the top K items to consider.
    
    Returns:
    - A tensor of shape (batch_size,) containing the MRR at K for each batch item.
    """
    ans = torch.stack(ans)
    
    k = min(k, bm_labels.size(1))
    bm_labels = bm_labels[:, :k]
    ans = ans[:, :k]

    # Get ranks of the correct answers
    correct = ans * (bm_labels == ans).float()
    ranks = torch.argmax(correct, dim=1) + 1
    ranks = ranks.float() * correct.max(dim=1)[0]  # Zero out ranks for misses

    # Calculate MRR
    mrr = 1.0 / ranks.clamp(min=1)
    mrr[ranks == 0] = 0.0  # Set MRR to 0 where there are no correct answers
    
    # Assuming calculate_mean_metric_tensor is implemented elsewhere
    return calculate_mean_metric_tensor(mrr)

def calculate_mean_metric_tensor(predictions):
    """
    Calculate the mean accuracy for a tensor of predictions.
    
    Parameters:
    - predictions: A tensor of binary predictions (0 or 1).
    
    Returns:
    - The mean accuracy as a tensor.
    """
    accuracy = predictions.float().mean()
    return accuracy
