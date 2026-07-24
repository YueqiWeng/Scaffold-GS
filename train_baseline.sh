#!/bin/bash
# Method 1/4: Scaffold-GS baseline (shared MLP, no MoE, no GNN)
# Output: outputs/<data>/baseline/<timestamp>/
# Usage:
#   bash train_baseline.sh -d mipnerf360/kitchen -g 0

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
logdir=baseline

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
    echo "Usage: bash train_baseline.sh -d <dataset> -g <gpu>"
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
  --num_experts 1 \
  ${add_opacity_dist:-} ${add_cov_dist:-} ${add_color_dist:-} \
  -m "${outdir}"
