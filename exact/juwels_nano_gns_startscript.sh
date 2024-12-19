#!/bin/bash
# general configuration of the job
#SBATCH --job-name=nanoGNS
#SBATCH --account=genai-ad
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=nanoGNS_%A_%a.out
#SBATCH --time=02:00:00

# configure node and process count on the CM
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --array=0-15

ml Python CUDA NCCL matplotlib

source nano_gpt_env_juwels/bin/activate

# Calculate derived parameters

# Create array of all combinations
declare -a combinations
index=0
for width in 256 512 1024 2048
do
    for lr in  0.015625 0.0078125 0.00390625 0.001953125
    do
        combinations[$index]="$width $lr"
        index=$((index + 1))
    done
done

# Get parameters for this array task
parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})
width=${parameters[0]}
lr=${parameters[1]}

head_size=64
n_heads=$((width / head_size))
min_lr=$(awk "BEGIN {print $lr/10}")
n_layer=12

gradient_accumulation_steps=8
batch_size=6
num_nodes=${SLURM_JOB_NUM_NODES}
# 1000 warmup steps
warmup_tokens=$((1024 * batch_size * gradient_accumulation_steps * num_nodes * 4 * 1000))

out_dir="/p/scratch/cslfse/aach1/mup_logs/gns_sp/layers_${n_layer}_width_${width}_lr_${lr}/"

echo "Running with layers=$n_layer, width=$width, lr=$lr, warmup_tokens=$warmup_tokens"

COMMAND="train.py --out_dir=$out_dir \
                --eval_interval=1000 \
                --log_interval=1 \
                --eval_iters=1 \
                --eval_only=False \
                --always_save_checkpoint=False \
                --init_from='scratch' \
                --wandb_log=False \
                --dataset="openwebtext" \
                --gradient_accumulation_steps=$gradient_accumulation_steps \
                --batch_size=$batch_size \
                --block_size=1024 \
                --n_layer=$n_layer \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=False \
                --lnclass='nn' \
                --learning_rate=$lr \
                --max_iters=10000 \
                --max_tokens=10**12 \
                --weight_decay=0.0 \
                --beta1=0.9 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --bs_schedule=False \
                --decay_lr=False \
                --warmup_tokens=0 \
                --lr_decay_tokens=0 \
                --min_lr=$min_lr \
                --backend='nccl' \
                --device='cuda' \
                --device_name='A100' \
                --dtype='bfloat16' \
                --compile=True \
                "

# sleep a sec
sleep 1

export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
export WANDB_MODE=offline

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=9469
export GPUS_PER_NODE=4

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"

srun bash -c "torchrun \
--nnodes=$SLURM_NNODES \
--rdzv_backend c10d --nproc_per_node=4 --rdzv_id $RANDOM --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
$COMMAND"

