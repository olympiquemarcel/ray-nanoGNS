#!/bin/bash
# general configuration of the job
#SBATCH --job-name=nanoGNS
#SBATCH --account=jhpc54
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=nanoGNS_test.out
#SBATCH --time=01:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

ml Python CUDA NCCL matplotlib

source nano_gpt_env/bin/activate

# Calculate derived parameters

width=256
lr=0.00390625
head_size=64
n_heads=$((width / head_size))
min_lr=$(awk "BEGIN {print $lr/10}")
out_dir="out/"
mup_base_width=256
mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)

echo "Running with width=$width, lr=$lr"

COMMAND="train.py --out_dir=$out_dir \
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

