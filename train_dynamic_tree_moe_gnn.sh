#!/bin/bash
# Method 4/4: Dynamic Tree MoE+GNN
#
# Hierarchy (required):
#   baseline → MoE → MoE+GNN → Dynamic
# Dynamic = MoE+GNN + growing expert tree. GNN is always on.
#
# Arch:
#   AnchorGNN → h_gnn every step
#   Dual router at every internal node (incl. root):
#     z = z_base(fv) + γ · z_gnn([fv ‖ h_gnn])
#   Leaf experts decode raw view-conditioned u
#   Selective leaf growth (grown internals also dual-GNN)
#
# Same Scaffold densify schedule as MoE / MoE+GNN (from scratch geometry).
# Optional warm-start: flat MoE MLP checkpoints → tree leaves (+ root base router).
#
# Usage:
#   bash train_dynamic_tree_moe_gnn.sh -d mipnerf360/kitchen -g 0

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
logdir=dynamic_tree_moe_gnn

# --- MoE+GNN + dynamic tree (canonical) ---
moe_arch=dynamic_tree
moe_branch_factor=4
moe_grow_factor=2
moe_tree_depth=3
moe_max_leaves=10
moe_max_total_grows=3
moe_top_k=2
moe_warmup=3000
lambda_load=0.005
lambda_router_entropy=0.002
lambda_opacity=0.01

gnn_layers=2
gnn_hidden_dim=32
gnn_k=16
gnn_gamma_init=0.5

tree_grow_start=8000
tree_grow_interval=4000
tree_grow_stop=20000
tree_grow_max_per_event=1
tree_grow_max_heads_per_event=1
tree_grow_min_score_ratio=1.15
tree_grow_min_neff=1.2
tree_grow_min_mass_frac=0.15
tree_grow_dense_iters=1500

start_baseline_mlp=""
warm_start_mode="auto"

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
        --moe_tree_depth) moe_tree_depth="$2"; shift ;;
        --moe_branch_factor) moe_branch_factor="$2"; shift ;;
        --moe_grow_factor) moe_grow_factor="$2"; shift ;;
        --moe_max_leaves) moe_max_leaves="$2"; shift ;;
        --moe_max_total_grows) moe_max_total_grows="$2"; shift ;;
        --gnn_gamma_init) gnn_gamma_init="$2"; shift ;;
        --tree_grow_start) tree_grow_start="$2"; shift ;;
        --tree_grow_interval) tree_grow_interval="$2"; shift ;;
        --tree_grow_stop) tree_grow_stop="$2"; shift ;;
        --tree_grow_max_heads_per_event) tree_grow_max_heads_per_event="$2"; shift ;;
        --tree_grow_min_score_ratio) tree_grow_min_score_ratio="$2"; shift ;;
        --tree_grow_min_neff) tree_grow_min_neff="$2"; shift ;;
        --start_baseline_mlp)
            start_baseline_mlp="$2"
            if [[ -z "$2" ]]; then warm_start_mode="off"; else warm_start_mode="path"; fi
            shift ;;
        --no_warm_start) warm_start_mode="off"; start_baseline_mlp="" ;;
        --add_opacity_dist) add_opacity_dist="--add_opacity_dist" ;;
        --add_cov_dist) add_cov_dist="--add_cov_dist" ;;
        --add_color_dist) add_color_dist="--add_color_dist" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$data" ]]; then
    echo "Usage: bash train_dynamic_tree_moe_gnn.sh -d <dataset> -g <gpu>"
    exit 1
fi

if [[ "$warm_start_mode" == "auto" ]]; then
    start_baseline_mlp="$(ls -td "outputs/${data}/moe/"*/point_cloud/iteration_30000/checkpoints.pth 2>/dev/null | head -1 || true)"
fi

time=$(date "+%Y-%m-%d_%H:%M:%S")
outdir="outputs/${data}/${logdir}/$time"

warm_args=()
if [[ -n "$start_baseline_mlp" && -e "$start_baseline_mlp" ]]; then
    warm_args=(--start_baseline_mlp "${start_baseline_mlp}")
    echo "[warm-start] MoE MLPs: ${start_baseline_mlp}"
else
    echo "[warm-start] disabled"
fi

echo "[dynamic] MoE+GNN+tree start=${moe_branch_factor} gamma=${gnn_gamma_init} max_grows=${moe_max_total_grows}"

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
  --moe_arch ${moe_arch} \
  --moe_branch_factor ${moe_branch_factor} \
  --moe_grow_factor ${moe_grow_factor} \
  --moe_tree_depth ${moe_tree_depth} \
  --moe_max_leaves ${moe_max_leaves} \
  --moe_max_total_grows ${moe_max_total_grows} \
  --moe_top_k ${moe_top_k} \
  --moe_warmup ${moe_warmup} \
  --lambda_opacity ${lambda_opacity} \
  --lambda_load ${lambda_load} \
  --lambda_router_entropy ${lambda_router_entropy} \
  --tree_grow_start ${tree_grow_start} \
  --tree_grow_interval ${tree_grow_interval} \
  --tree_grow_stop ${tree_grow_stop} \
  --tree_grow_max_per_event ${tree_grow_max_per_event} \
  --tree_grow_max_heads_per_event ${tree_grow_max_heads_per_event} \
  --tree_grow_min_score_ratio ${tree_grow_min_score_ratio} \
  --tree_grow_min_neff ${tree_grow_min_neff} \
  --tree_grow_min_mass_frac ${tree_grow_min_mass_frac} \
  --tree_grow_dense_iters ${tree_grow_dense_iters} \
  --use_gnn \
  --gnn_layers ${gnn_layers} \
  --gnn_hidden_dim ${gnn_hidden_dim} \
  --gnn_k ${gnn_k} \
  --gnn_gamma_init ${gnn_gamma_init} \
  "${warm_args[@]}" \
  ${add_opacity_dist:-} ${add_cov_dist:-} ${add_color_dist:-} \
  -m "${outdir}"
