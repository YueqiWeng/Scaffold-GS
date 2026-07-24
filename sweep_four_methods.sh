#!/usr/bin/env bash
# Sweep the 4 methods across dataset suites (sequential, single GPU by default).
#
# Methods (in order per scene so dynamic can warm-start from flat MoE):
#   1) baseline
#   2) moe              (flat MoE)
#   3) moe_gnn
#   4) dynamic_tree_moe_gnn
#
# Usage examples:
#   bash sweep_four_methods.sh --suite mip360 -g 0
#   bash sweep_four_methods.sh --suite tandt,db -g 0
#   bash sweep_four_methods.sh --suite all -g 0
#   bash sweep_four_methods.sh --scenes mipnerf360/kitchen,tandt/truck -g 0
#   bash sweep_four_methods.sh --suite mip360 --methods baseline,moe -g 0 --dry-run
#   bash sweep_four_methods.sh --suite mip360 --skip-done -g 0
#
# Logs: sweep_logs/sweep_<timestamp>.log
# Summary table written at the end (and on Ctrl-C via trap if possible).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU="${GPU:-0}"
SUITE="mip360"
SCENES_CSV=""
METHODS_CSV="baseline,moe,moe_gnn,dynamic"
SKIP_DONE=0
DRY_RUN=0
CONTINUE_ON_ERROR=1
ITERATIONS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpu) GPU="$2"; shift ;;
        --suite) SUITE="$2"; shift ;;
        --scenes) SCENES_CSV="$2"; shift ;;
        --methods) METHODS_CSV="$2"; shift ;;
        --skip-done) SKIP_DONE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --stop-on-error) CONTINUE_ON_ERROR=0 ;;
        --iterations) ITERATIONS="$2"; shift ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# conda
if [[ "${CONDA_DEFAULT_ENV:-}" != "scaffold_gs" ]]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scaffold_gs
    set -u
fi
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP 2>/dev/null || true
export CUDA_HOME="${CONDA_PREFIX:-}"
export PATH="${CONDA_PREFIX:-}/bin:${PATH}"

mkdir -p sweep_logs
STAMP="$(date '+%Y-%m-%d_%H:%M:%S')"
LOG="sweep_logs/sweep_${STAMP}.log"
SUMMARY="sweep_logs/summary_${STAMP}.txt"

log() { echo "$@" | tee -a "$LOG"; }

scene_exists() {
    local s="$1"
    [[ -d "data/$s/sparse" || -f "data/$s/transforms_train.json" || -d "data/$s/images" ]]
}

# dataset family → (voxel_size, update_init_factor, lod)
scene_hparams() {
    local s="$1"
    case "$s" in
        mipnerf360/*)      echo "0.001 16 0" ;;
        tandt/*)           echo "0.01 16 0" ;;
        db/*|blending/*)   echo "0.005 16 0" ;;
        bungeenerf/*)      echo "0 128 30" ;;
        nerfsynthetic/*)   echo "0.001 4 0" ;;
        *)                 echo "0.001 16 0" ;;
    esac
}

expand_suite() {
    local suite="$1"
    case "$suite" in
        mip360|mipnerf360)
            echo "mipnerf360/bicycle mipnerf360/bonsai mipnerf360/counter mipnerf360/garden mipnerf360/kitchen mipnerf360/room mipnerf360/stump"
            ;;
        tandt|tnt)
            echo "tandt/train tandt/truck"
            ;;
        db|blending)
            echo "db/drjohnson db/playroom"
            ;;
        bungee|bungeenerf)
            echo "bungeenerf/amsterdam bungeenerf/bilbao bungeenerf/hollywood bungeenerf/pompidou bungeenerf/quebec bungeenerf/rome bungeenerf/barcelona bungeenerf/chicago"
            ;;
        nerf|nerfsynthetic)
            echo "nerfsynthetic/chair nerfsynthetic/drums nerfsynthetic/ficus nerfsynthetic/hotdog nerfsynthetic/lego nerfsynthetic/materials nerfsynthetic/mic nerfsynthetic/ship"
            ;;
        paper|all)
            # full Scaffold-GS paper suite
            echo "$(expand_suite mip360) $(expand_suite tandt) $(expand_suite db) $(expand_suite bungee) $(expand_suite nerf)"
            ;;
        quick)
            # small multi-dataset smoke set
            echo "mipnerf360/kitchen tandt/truck db/playroom nerfsynthetic/lego"
            ;;
        *)
            echo "Unknown suite: $suite" >&2
            exit 1
            ;;
    esac
}

# Build scene list
SCENES=()
if [[ -n "$SCENES_CSV" ]]; then
    IFS=',' read -r -a SCENES <<< "$SCENES_CSV"
else
    IFS=',' read -r -a SUITES <<< "$SUITE"
    for su in "${SUITES[@]}"; do
        # shellcheck disable=SC2207
        SCENES+=($(expand_suite "$su"))
    done
fi

IFS=',' read -r -a METHODS <<< "$METHODS_CSV"

method_done() {
    local scene="$1" method="$2"
    local dir="outputs/${scene}/${method}"
    [[ -d "$dir" ]] || return 1
    find "$dir" -name results.json 2>/dev/null | grep -q .
}

latest_moe_ckpt() {
    local scene="$1"
    ls -td "outputs/${scene}/moe/"*/point_cloud/iteration_30000/checkpoints.pth 2>/dev/null | head -1 || true
}

run_one() {
    local scene="$1" method="$2"
    local hp vsize init_factor lod
    hp="$(scene_hparams "$scene")"
    read -r vsize init_factor lod <<< "$hp"

    local extra=( -d "$scene" -g "$GPU" --voxel_size "$vsize" --update_init_factor "$init_factor" --lod "$lod" )
    if [[ -n "$ITERATIONS" ]]; then
        extra+=( --iterations "$ITERATIONS" )
    fi

    local script=""
    case "$method" in
        baseline) script="train_baseline.sh" ;;
        moe) script="train_moe.sh" ;;
        moe_gnn) script="train_moe_gnn.sh" ;;
        dynamic|dynamic_tree|dynamic_tree_moe_gnn)
            method="dynamic_tree_moe_gnn"
            script="train_dynamic_tree_moe_gnn.sh"
            local ckpt
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -n "$ckpt" ]]; then
                extra+=( --start_baseline_mlp "$ckpt" )
            else
                extra+=( --no_warm_start )
                log "[warn] no flat MoE ckpt for $scene; dynamic runs without warm-start"
            fi
            ;;
        *)
            log "[error] unknown method: $method"
            return 1
            ;;
    esac

    if [[ "$SKIP_DONE" -eq 1 ]] && method_done "$scene" "$method"; then
        log "[skip-done] $scene / $method"
        return 0
    fi

    log ""
    log "========================================"
    log "[$(date '+%F %T')] $method  on  $scene  (voxel=$vsize init=$init_factor lod=$lod)"
    log "========================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "DRY: bash $script ${extra[*]}"
        return 0
    fi

    set +e
    bash "$script" "${extra[@]}" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        log "[FAIL] $method on $scene  exit=$rc"
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            return "$rc"
        fi
    else
        log "[OK] $method on $scene"
    fi
    return 0
}

write_summary() {
    python3 - <<'PY' "$SUMMARY" | tee -a "$LOG"
import json, glob, os, sys
out_path = sys.argv[1]
methods = ["baseline", "moe", "moe_gnn", "dynamic_tree_moe_gnn"]
rows = []
for p in sorted(glob.glob("outputs/**/results.json", recursive=True)):
    parts = p.split(os.sep)
    # outputs/<suite>/<scene>/<method>/<timestamp>/results.json  OR
    # outputs/<suite/scene>/<method>/<timestamp>/results.json
    try:
        idx = parts.index("outputs")
        rest = parts[idx+1:]
        # find method folder name
        method = None
        for m in methods:
            if m in rest:
                method = m
                mi = rest.index(m)
                scene = "/".join(rest[:mi])
                break
        if method is None:
            continue
        with open(p) as f:
            d = json.load(f)
        # prefer ours_30000
        key = "ours_30000" if "ours_30000" in d else next(iter(d))
        m = d[key]
        rows.append((scene, method, float(m["PSNR"]), float(m["SSIM"]), float(m["LPIPS"]), p))
    except Exception:
        continue

# keep latest per (scene, method) by path sort (timestamp in path)
best = {}
for scene, method, psnr, ssim, lpips, p in rows:
    k = (scene, method)
    if k not in best or p > best[k][-1]:
        best[k] = (psnr, ssim, lpips, p)

scenes = sorted({k[0] for k in best})
lines = []
lines.append("")
lines.append("=== Sweep summary (latest results.json per scene/method) ===")
hdr = f"{'scene':40s} {'baseline':>8s} {'moe':>8s} {'moe_gnn':>8s} {'dynamic':>8s}"
lines.append(hdr)
lines.append("-" * len(hdr))
for scene in scenes:
    vals = []
    for m in methods:
        if (scene, m) in best:
            vals.append(f"{best[(scene, m)][0]:8.3f}")
        else:
            vals.append(f"{'—':>8s}")
    lines.append(f"{scene:40s} " + " ".join(vals))
lines.append("")
lines.append("PSNR columns: baseline | flat moe | moe+gnn | dynamic beat-flat")
text = "\n".join(lines)
print(text)
with open(out_path, "w") as f:
    f.write(text + "\n")
PY
}

log "Sweep started at $(date)"
log "GPU=$GPU  suite=$SUITE  scenes=${SCENES[*]}"
log "methods=${METHODS[*]}  skip_done=$SKIP_DONE  dry_run=$DRY_RUN"
log "log=$LOG"

trap 'log "[trap] writing summary..."; write_summary || true' EXIT

for scene in "${SCENES[@]}"; do
    if ! scene_exists "$scene"; then
        log "[SKIP missing data] $scene"
        continue
    fi
    for method in "${METHODS[@]}"; do
        run_one "$scene" "$method" || true
    done
done

log ""
log "Sweep finished at $(date)"
write_summary
log "Summary: $SUMMARY"
