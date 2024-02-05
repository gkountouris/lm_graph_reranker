import logging
import os

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import modeling_bert
    from transformers import modeling_roberta
except:
    from transformers.models.bert import modeling_bert
    from transformers.models.roberta import modeling_roberta
from transformers import PretrainedConfig
from transformers.file_utils import (
     TF2_WEIGHTS_NAME,
     TF_WEIGHTS_NAME,
     WEIGHTS_NAME,
     cached_path,
     hf_bucket_url,
     is_remote_url,
)

import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter

print('hello world')
print(torch.__version__ >= '1.6.0')