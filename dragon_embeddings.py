import argparse
import logging
import random
import shutil
import time
import json

# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb

from modeling import modeling_dragon
from utils import data_utils
from utils import parser_utils
from utils import utils

import numpy as np

import socket, os, sys, subprocess

logger = logging.getLogger(__name__)

def tensor_memory_size(tensor):
    """
    Calculate the memory size occupied by a tensor.
    """
    # Get number of elements in tensor
    num_elements = tensor.numel()
    
    # Get size of each element in bytes
    element_size = tensor.element_size()
    
    # Total memory in bytes
    total_bytes = num_elements * element_size
    
    # Convert bytes to kilobytes (1 KB = 1024 Bytes)
    total_kilobytes = total_bytes / 1024
    
    # Convert kilobytes to megabytes (1 MB = 1024 KB)
    total_megabytes = total_kilobytes / 1024

    total_gigabytes = total_megabytes / 1024
    
    return total_gigabytes

def load_data(args, devices, kg):
    _seed = args.seed
    if args.local_rank != -1:
        _seed = args.seed + (2** args.local_rank -1) #use different seed for different gpu process so that they have different training examples
    print ('_seed', _seed, file=sys.stderr)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(_seed)

    #########################################################
    # Construct the dataset
    #########################################################
    one_process_at_a_time = args.data_loader_one_process_at_a_time

    if args.local_rank != -1 and one_process_at_a_time:
        for p_rank in range(args.world_size):
            if args.local_rank != p_rank: # Barrier
                torch.distributed.barrier()
            dataset = data_utils.DRAGON_DataLoader(args, args.train_statements, args.train_adj,
                args.dev_statements, args.dev_adj,
                args.test_statements, args.test_adj,
                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                device=devices,
                model_name=args.encoder,
                max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)
            if args.local_rank == p_rank: #End of barrier
                torch.distributed.barrier()
    else:
        dataset = data_utils.DRAGON_DataLoader(args, args.train_statements, args.train_adj,
            args.dev_statements, args.dev_adj,
            args.test_statements, args.test_adj,
            batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            device=devices,
            model_name=args.encoder,
            max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
            is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
            subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset

def construct_model(args, kg, dataset):
    ########################################################
    #   Load pretrained concept embeddings
    ########################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    ##########################################################
    #   Build model
    ##########################################################

    if kg == "umls":
        n_ntype = 4
        n_etype = dataset.final_num_relation *2
        print ('final_num_relation', dataset.final_num_relation, 'len(id2relation)', len(dataset.id2relation))
        print ('final_num_relation', dataset.final_num_relation, 'len(id2relation)', len(dataset.id2relation), file=sys.stderr)
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    print ('n_ntype', n_ntype, 'n_etype', n_etype)
    print ('n_ntype', n_ntype, 'n_etype', n_etype, file=sys.stderr)
    encoder_load_path = args.encoder_load_path if args.encoder_load_path else args.encoder
    model = modeling_dragon.DRAGON(args, encoder_load_path, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
        concept_dim=args.gnn_dim,
        concept_in_dim=concept_in_dim,
        n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
        p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
        pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
        init_range=args.init_range, ie_dim=args.ie_dim, info_exchange=args.info_exchange, ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers, layer_id=args.encoder_layer)
    return model

def sep_params(model, loaded_roberta_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_roberta_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params

def count_parameters(loaded_params, not_loaded_params):
    num_params = sum(p.numel() for p in not_loaded_params.values() if p.requires_grad)
    num_fixed_params = sum(p.numel() for p in not_loaded_params.values() if not p.requires_grad)
    num_loaded_params = sum(p.numel() for p in loaded_params.values())
    print('num_trainable_params (out of not_loaded_params):', num_params)
    print('num_fixed_params (out of not_loaded_params):', num_fixed_params)
    print('num_loaded_params:', num_loaded_params)
    print('num_total_params:', num_params + num_fixed_params + num_loaded_params)

def calc_loss_and_acc(logits, labels, loss_type, loss_func):
    if logits is None:
        loss = 0.
        n_corrects = 0
    else:
        if loss_type == 'margin_rank':
            raise NotImplementedError
        elif loss_type == 'cross_entropy':
            loss = loss_func(logits, labels)
        bs = labels.size(0)
        loss *= bs
        n_corrects = (logits.argmax(1) == labels).sum().item()

    return loss, n_corrects

def calc_eval_accuracy(args, eval_set, model, loss_type, loss_func, debug, save_test_preds, preds_path):
    """Eval on the dev or test set - calculate loss and accuracy"""
    total_loss_acm = end_loss_acm = mlm_loss_acm = 0.0
    link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    model.eval()
    save_test_preds = (save_test_preds and args.end_task)
    if save_test_preds:
        utils.check_path(preds_path)
        f_preds = open(preds_path, 'w')
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set, desc="Dev/Test batch"):
            bs = labels.size(0)
            logits, mlm_loss, link_losses = model(*input_data)
            end_loss, n_corrects = calc_loss_and_acc(logits, labels, loss_type, loss_func)
            link_loss, pos_link_loss, neg_link_loss = link_losses
            loss = args.end_task * end_loss + args.mlm_task * mlm_loss + args.link_task * link_loss

            total_loss_acm += float(loss)
            end_loss_acm += float(end_loss)
            mlm_loss_acm += float(mlm_loss)
            link_loss_acm += float(link_loss)
            pos_link_loss_acm += float(pos_link_loss)
            neg_link_loss_acm += float(neg_link_loss)
            n_corrects_acm += n_corrects
            n_samples_acm += bs

            if save_test_preds:
                predictions = logits.argmax(1) #[bsize, ]
                for qid, pred in zip(qids, predictions):
                    print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                    f_preds.flush()
            if debug:
                break
    if save_test_preds:
        f_preds.close()
    total_loss_avg, end_loss_avg, mlm_loss_avg, link_loss_avg, pos_link_loss_avg, neg_link_loss_avg, n_corrects_avg = \
        [item / n_samples_acm for item in (total_loss_acm, end_loss_acm, mlm_loss_acm, link_loss_acm, pos_link_loss_acm, neg_link_loss_acm, n_corrects_acm)]
    return total_loss_avg, end_loss_avg, mlm_loss_avg, link_loss_avg, pos_link_loss_avg, neg_link_loss_avg, n_corrects_avg

def calc_embeddings(args, eval_set, model, debug):
    """On set - calculate embeddings"""
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set, desc="Docs batch"):
            bs = labels.size(0)
            logits, mlm_loss, link_losses = model(*input_data)
            print(logits)
            break
    return 

def embeddings(args, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse

    # args = utils.import_config(checkpoint["config"], args)
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse

    print('HEYYYYYYYYYYYYYYYY')
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    model = construct_model(args, kg, dataset)
    INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
    bert_or_roberta = model.lmgnn.bert if INHERIT_BERT else model.lmgnn.roberta
    bert_or_roberta.resize_token_embeddings(len(dataset.tokenizer))

    model.load_state_dict(checkpoint["model"], strict=False)
    epoch_id = checkpoint.get('epoch', 0)

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    model.eval()

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    model.eval()
    # Evaluation on the dev set
    preds_path = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
    
    dev_total_loss, dev_end_loss, dev_mlm_loss, dev_link_loss, dev_pos_link_loss, dev_neg_link_loss, dev_acc  = calc_embeddings(args, dev_dataloader, model, not debug)

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

def run_dragon(args, resume, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse

    # args = utils.import_config(checkpoint["config"], args)
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse

    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg, dataset)
    INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
    bert_or_roberta = model.lmgnn.bert if INHERIT_BERT else model.lmgnn.roberta
    bert_or_roberta.resize_token_embeddings(len(dataset.tokenizer))

    model.load_state_dict(checkpoint["model"], strict=False)
    epoch_id = checkpoint.get('epoch', 0)

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    model.eval()

def main(args):
    
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    has_test_split = False
    devices = get_devices(args)

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.resume_checkpoint not in [None, "None"]

    args.hf_version = transformers.__version__

    if args.local_rank in [-1, 0]:
        wandb_id = args.resume_id if resume and (args.resume_id not in [None, "None"]) else wandb.util.generate_id()
        args.wandb_id = wandb_id
        wandb.init(project="DRAGON", config=args, name=args.run_name, resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode)
        print(socket.gethostname())
        print ("pid:", os.getpid())
        print ("conda env:", os.environ.get('CONDA_DEFAULT_ENV'))
        print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
        print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
        utils.print_cuda_info()
        print("wandb id: ", wandb_id)
        wandb.run.log_code('.')

    kg = args.kg
    if args.dataset == "medqa_usmle":
        kg = "ddb"
    elif args.dataset in ["medqa", "pubmedqa", "bioasq"]:
        kg = "umls"
    print ("KG used:", kg)
    print ("KG used:", kg, file=sys.stderr)

    if args.mode == 'embeddings':
        run_dragon(args, resume, has_test_split, devices, kg)
        # assert args.world_size == 1, "DDP is only implemented for training"
        # embeddings(args, devices, kg)
    else:
        raise ValueError('Invalid mode')
    


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='embeddings', choices=['embeddings', 'train', 'eval'], help='run training or evaluation')
    parser.add_argument('--load_model_path', default='./models/biomed_model.pt', help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")
    parser.add_argument("--load_graph_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--dump_graph_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")
    parser.add_argument("--data_loader_one_process_at_a_time", default=False, type=utils.bool_flag)

    #Task
    parser.add_argument('--end_task', type=float, default=0.0, help='Task weight for the end task (MCQA)')
    parser.add_argument('--mlm_task', type=float, default=0.0, help='Task weight for the MLM task')
    parser.add_argument('--link_task', type=float, default=0.0, help='Task weight for the LinkPred task')

    # Data
    parser.add_argument('--kg', default='umls', help="What KG to use.")
    parser.add_argument('--max_num_relation', default=-1, type=int, help="max number of KG relation types to keep.")
    parser.add_argument('--kg_only_use_qa_nodes', default=False, type=utils.bool_flag, help="")

    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of Fusion layers')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--residual_ie', default=0, type=int, help='Whether to use residual MInt.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every Fusion layer or every other Fusion layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt Fusion layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")
    parser.add_argument('--no_node_score', default=True, type=utils.bool_flag, help='Don\'t use node score.')

    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=1e-3, type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int, help="Number of the first few epochs in which LM's parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--fp16', default=False, type=utils.bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--upcast', default=False, type=utils.bool_flag, help='Upcast attention computation during fp16 training')
    parser.add_argument('--redef_epoch_steps', default=-1, type=int)

    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    if args.local_rank != -1:
        assert not args.dump_graph_cache
    main(args)
