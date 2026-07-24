#!/bin/bash
# Method 3/4: Flat MoE + dual-router GNN. Output: outputs/<data>/moe_gnn/<timestamp>/
# Arch: BaseRouter(fv) + γ * GnnRouter(attention kNN context) → Top-k → experts(fv,view)
# Usage:
#   bash train_moe_gnn.sh -d mipnerf360/kitchen -g 0

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

port=$(rand 10000 30000)
lod=0
iterations=30_000
appearance_dim=0
voxel_size=0.001
update_init_factor=16
ratio=1
gpu=0
data=""
logdir=moe_gnn

# Identical MoE recipe to train_moe.sh
num_experts=8
moe_top_k=2
moe_warmup=3000
lambda_load=0.005
lambda_router_entropy=0.002
lambda_opacity=0.01
gnn_layers=2
gnn_hidden_dim=32
gnn_k=16

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--data) data="$2"; shift ;;
        -g|--gpu) gpu="$2"; shift ;;
        -l|--logdir) logdir="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --iterations) iterations="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --voxel_size) voxel_size="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --add_opacity_dist) add_opacity_dist="--add_opacity_dist" ;;
        --add_cov_dist) add_cov_dist="--add_cov_dist" ;;
        --add_color_dist) add_color_dist="--add_color_dist" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$data" ]]; then
    echo "Usage: bash train_moe_gnn.sh -d <dataset> -g <gpu>"
    exit 1
fi

time=$(date "+%Y-%m-%d_%H:%M:%S")
outdir="outputs/${data}/${logdir}/$time"

python train.py --eval \
  -s "data/${data}" \
  --lod ${lod} \
  --gpu ${gpu} \
  --voxel_size ${voxel_size} \
  --update_init_factor ${update_init_factor} \
  --appearance_dim ${appearance_dim} \
  --ratio ${ratio} \
  --iterations ${iterations} \
  --port ${port} \
  --num_experts ${num_experts} \
  --moe_top_k ${moe_top_k} \
  --moe_warmup ${moe_warmup} \
  --lambda_opacity ${lambda_opacity} \
  --lambda_load ${lambda_load} \
  --lambda_router_entropy ${lambda_router_entropy} \
  --use_gnn \
  --gnn_layers ${gnn_layers} \
  --gnn_hidden_dim ${gnn_hidden_dim} \
  --gnn_k ${gnn_k} \
  ${add_opacity_dist:-} ${add_cov_dist:-} ${add_color_dist:-} \
  -m "${outdir}"
