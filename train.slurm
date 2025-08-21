#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# --- Training Configuration Arguments ---
N_GPUS=4
DATASET='fineweb'
TOTAL_BATCH_SIZE_STR="2**14"
BATCH_SIZE=2
MAX_ITERS=150000
LEARNING_RATE=7e-5
WARMUP_STEPS=500
GRAD_CLIP=0.9
EVAL=true
EVAL_INTERVAL=100
EVAL_ITERS=10
SAVE_MODEL=true
FILE_NAME="llm_model"
ACT_RECOMP=true

# --- Model Configuration Arguments ---
N_LAYER=12
N_EMBD=1024
VOCAB_SIZE=50304
BLOCK_SIZE=1024
DROPOUT=0.01
POS_EMB="rope"

UP_DIM=768
NON_LINEARITY="swiglu"

ATTN="mla"
N_HEAD=8
N_KV_HEADS=4
Q_LATENT_DIM=256
KV_LATENT_DIM=256
ROPE_HEAD_DIM=128

MOE=true
N_EXP=16
N_SHARED=1
N_ACT=4
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
CEOFF=0.01

# Run training with torchrun (single-node, multi-GPU)
torchrun --standalone --nproc_per_node=$N_GPUS \
    train.py \
    --dataset $DATASET \
    --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --eval_interval $EVAL_INTERVAL \
    --eval_iters $EVAL_ITERS \
    --n_layer $N_LAYER \
    --n_embd $N_EMBD \
    --vocab_size $VOCAB_SIZE \
    --block_size $BLOCK_SIZE \
    --dropout $DROPOUT \
    --pos_emb $POS_EMB \
    --up_dim $UP_DIM \
    --non_linearity $NON_LINEARITY \
    --attn $ATTN \
    --n_head $N_HEAD \
    --n_kv_heads $N_KV_HEADS \
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --n_exp $N_EXP \
    --n_shared $N_SHARED \
    --n_act $N_ACT \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --coeff $CEOFF \
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$MOE" = true ] && echo "--moe" ) \
    $( [ "$ACT_RECOMP" = true ] && echo "--act_recomp" ) \
    $( [ "$AUX_FREE" = true ] && echo "--aux_free" )
