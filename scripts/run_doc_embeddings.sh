#!/bin/bash

# export LD_LIBRARY_PATH=/storage3/gkou/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2
export DEVICE_TO_USE="0,1,2"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export INHERIT_BERT=1
dt=`date '+%Y%m%d_%H%M%S'`


dataset="pubmedqa"
shift
encoder='michiyasunaga/BioLinkBERT-large'
# load_model_path=None
load_model_path=models/biomed_model.pt
args=$@

#loss function
custom_rank_loss=custom_rank_loss


elr="2e-5"
dlr="1e-4"
bs=32
mbs=4
unfreeze_epoch=0
k=5 #num of gnn layers
residual_ie=2
gnndim=200


encoder_layer=-1
max_node_num=400
seed=5
lr_schedule=warmup_linear
warmup_steps=500

n_epochs=3
max_epochs_before_stop=100
ie_dim=400
max_rank=1000

max_seq_len=512
ent_emb=data/umls/ent_emb_blbertL.npy
# tf_idf_path=data/pubmed/sparse_matrix.npz
tf_idf_path=data/pubmedqa/tensors
kg=umls
kg_vocab_path=data/umls/concepts.txt
inhouse=false


info_exchange=true
ie_layer_num=1
resume_checkpoint=None
resume_id=None
sep_ie_layers=false
random_ent_emb=false

fp16=false
upcast=true

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "ie_dim: ${ie_dim}, info_exchange: ${info_exchange}"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs

run_name=drums_rerank__${dataset}_ih_${inhouse}_load__elr${elr}_dlr${dlr}_b${bs}_ufz${unfreeze_epoch}_e${n_epochs}_sd${seed}__${dt}
# run_name=drums__${dataset}_ih_${inhouse}_load__elr${elr}_dlr${dlr}_b${bs}_ufz${unfreeze_epoch}_e${n_epochs}_sd${seed}__${dt}

log=logs/train__${run_name}.log.txt

###### Training ######
python3 -u drums_rerank.py --mode train \
    --dataset $dataset \
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --max_rank $max_rank --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --fp16 $fp16 --upcast $upcast --use_wandb false \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --save_model 0 \
    --run_name ${run_name} \
    --load_model_path $load_model_path \
    --residual_ie $residual_ie \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb_paths ${ent_emb//,/ } --lr_schedule ${lr_schedule} --warmup_steps $warmup_steps -ih ${inhouse} --kg $kg --kg_vocab_path $kg_vocab_path --tf_idf_path $tf_idf_path \
    --data_dir data \
    --loss $custom_rank_loss \
> ${log}
