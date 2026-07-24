#!/usr/bin/env bash
# Train all Scaffold-GS paper scenes sequentially on a single GPU (~6-8 hours on RTX 5080)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# conda activate can trip `set -u` on CONDA_BACKUP_CXX; relax nounset briefly
if [[ "${CONDA_DEFAULT_ENV:-}" != "scaffold_gs" ]]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scaffold_gs
    set -u
fi

# Required for CUDA extension builds if re-run; harmless during training
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP 2>/dev/null || true
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

EXP_NAME="baseline"
GPU="${GPU:-0}"   # use GPU 0; set GPU=-1 for auto-pick idle GPU
LOG="reproduce_all.log"

train_scene() {
    local scene="$1" vsize="$2" init_factor="$3" lod="${4:-0}" warmup="${5:-False}" exp="${6:-$EXP_NAME}"
    echo ""
    echo "========================================"
    echo "Training: $scene  (voxel=$vsize init=$init_factor lod=$lod warmup=$warmup)"
    echo "========================================"
    ./train.sh -d "$scene" -l "$exp" --gpu "$GPU" \
        --voxel_size "$vsize" --update_init_factor "$init_factor" \
        --appearance_dim 0 --ratio 1 --lod "$lod" --warmup "$warmup" \
        2>&1 | tee -a "$LOG"
}

echo "Full reproduction started at $(date)" | tee "$LOG"

# --- Tanks&Temples (voxel_size=0.01) ---
for s in tandt/train tandt/truck; do
    [[ -d "data/$s/sparse" ]] && train_scene "$s" 0.01 16 || echo "SKIP missing: $s"
done

# --- Deep Blending (voxel_size=0.005) ---
for s in db/drjohnson db/playroom; do
    [[ -d "data/$s/sparse" ]] && train_scene "$s" 0.005 16 || echo "SKIP missing: $s"
done

# --- MipNeRF360 (voxel_size=0.001) ---
for s in bicycle bonsai counter garden kitchen room stump; do
    scene="mipnerf360/$s"
    if [[ -d "data/$scene/sparse" || -f "data/$scene/transforms_train.json" ]]; then
        train_scene "$scene" 0.001 16
    else
        echo "SKIP missing: $scene"
    fi
done

# --- BungeeNeRF (voxel_size=0, lod=30) ---
for s in amsterdam bilbao hollywood pompidou quebec rome barcelona chicago; do
    scene="bungeenerf/$s"
    if [[ -d "data/$scene/sparse" ]]; then
        train_scene "$scene" 0 128 30
    else
        echo "SKIP missing: $scene"
    fi
done

# --- NeRF Synthetic without warmup (voxel_size=0.001) ---
for s in chair drums ficus hotdog lego materials mic ship; do
    scene="nerfsynthetic/$s"
    if [[ -f "data/$scene/transforms_train.json" ]]; then
        train_scene "$scene" 0.001 4
    else
        echo "SKIP missing: $scene"
    fi
done

echo ""
echo "========================================"
echo "Reproduction finished at $(date)"
echo "Results in: outputs/"
echo "Summary log: $LOG"
echo "========================================"

# Collect metrics into one file
python3 - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("outputs/**/results.json", recursive=True)):
    try:
        with open(p) as f:
            m = json.load(f)
        rows.append((p, m))
    except Exception:
        pass
if rows:
    print("\n=== Metrics summary ===")
    for p, m in rows:
        scene = p.split("outputs/")[1].rsplit("/", 2)[0]
        psnr = m.get("PSNR", m.get("psnr", "?"))
        ssim = m.get("SSIM", m.get("ssim", "?"))
        lpips = m.get("LPIPS", m.get("lpips", "?"))
        print(f"{scene:45s}  PSNR={psnr}  SSIM={ssim}  LPIPS={lpips}")
PY
