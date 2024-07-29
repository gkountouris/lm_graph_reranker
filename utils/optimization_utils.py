import math
import torch
from transformers import AdamW
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import sys


class SoftRankCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftRankCrossEntropyLoss, self).__init__()

    def forward(self, top_scores, true_label):

        # Ensure the true_label is an integer for indexing
        true_label_idx = true_label.item() if isinstance(true_label, torch.Tensor) else true_label
        
        # try:     
        # Get the score corresponding to the true label
        true_label_score = top_scores[true_label_idx]

        # Calculate negative log likelihood for the true label
        loss = -torch.log(true_label_score + 1e-10)  # Adding epsilon for numerical stability

        return loss


class CustomRankLoss(nn.Module):
    def __init__(self, max_rank=1000):
        super(CustomRankLoss, self).__init__()
        self.max_rank = max_rank

    def forward(self, top_indices_list, true_labels):
        total_loss = 0.0

        for top_indices, true_label in zip(top_indices_list, true_labels):
            # Find if the true label is in top_indices and its rank
            matches = top_indices == true_label.item()
            rank = matches.nonzero(as_tuple=True)[0]

            if len(rank) == 0:
                # True label not found in top_indices, assign maximum loss
                loss = 1.0
            else:
                # Normalize the rank to be between 0 and 1
                loss = rank.float().item() / self.max_rank

            total_loss += loss

        # Average the loss over the batch
        average_loss = total_loss / len(true_labels)

        return torch.tensor(average_loss, device=true_labels.device)


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # # more conservative since it's an approximated value
                # if N_sma >= 5:
                #     if group['weight_decay'] != 0:
                #         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                #     denom = exp_avg_sq.sqrt().add_(group['eps'])
                #     p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                #     p.data.copy_(p_data_fp32)
                # elif step_size > 0:
                #     if group['weight_decay'] != 0:
                #         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                #     p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                #     p.data.copy_(p_data_fp32)

                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        # Corrected weight decay application
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        # Corrected weight decay application
                        p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss


OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}


def run_test():
    import torch.nn as nn
    model = nn.Sequential(*[nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 2)])
    x = torch.randn(10, 100).repeat(100, 1)
    y = torch.randint(0, 2, (10,)).repeat(100)
    crit = nn.CrossEntropyLoss()
    optim = RAdam(model.parameters(), lr=1e-2, weight_decay=0.01)
    model.train()
    for a in range(0, 1000, 10):
        b = a + 10
        loss = crit(model(x[a:b]), y[a:b])
        loss.backward()
        optim.step()
        print('| loss: {:.4f} |'.format(loss.item()))


if __name__ == '__main__':
    run_test()
