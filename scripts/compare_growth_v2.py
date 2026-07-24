#!/usr/bin/env python3
"""Compare growth-policy v2 dynamic runs vs flat MoE and old dynamic (Mip360 garden/stump)."""
import json
import glob
import os
import re

METHODS = {
    "moe": "flat MoE",
    "dynamic_tree_moe_gnn": "dynamic",
}
SCENES = ["mipnerf360/garden", "mipnerf360/stump"]
method_re = re.compile(r"/(moe_gnn|dynamic_tree_moe_gnn|moe)/")


def best_results(scene: str):
    out = {}
    for p in glob.glob(f"outputs/{scene}/**/results.json", recursive=True):
        m = method_re.search("/" + p.replace("\\", "/"))
        if not m:
            continue
        method = m.group(1)
        if method == "moe_gnn":
            continue
        parts = p.split(os.sep)
        # outputs / scene... / method / stamp / results.json
        try:
            mi = parts.index(method)
            stamp = parts[mi + 1]
        except (ValueError, IndexError):
            continue
        with open(p) as f:
            d = json.load(f)
        key = "ours_30000" if "ours_30000" in d else next(iter(d))
        row = {k: float(v) for k, v in d[key].items() if isinstance(v, (int, float))}
        if method not in out or stamp > out[method]["stamp"]:
            out[method] = {"stamp": stamp, "metrics": row, "path": p}
    return out


def main():
    print(f"{'scene':10s} {'run':28s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}  stamp")
    for scene in SCENES:
        name = scene.split("/")[-1]
        best = best_results(scene)
        # list all dynamic stamps for v1 vs v2
        dyn_runs = []
        for p in glob.glob(f"outputs/{scene}/dynamic_tree_moe_gnn/*/results.json"):
            stamp = p.split(os.sep)[-2]
            with open(p) as f:
                d = json.load(f)
            key = "ours_30000" if "ours_30000" in d else next(iter(d))
            m = d[key]
            dyn_runs.append((stamp, float(m["PSNR"]), float(m["SSIM"]), float(m["LPIPS"])))
        dyn_runs.sort()

        moe = best.get("moe")
        if moe:
            m = moe["metrics"]
            print(
                f"{name:10s} {'flat MoE':28s} {m['PSNR']:7.3f} {m['SSIM']:7.4f} {m['LPIPS']:7.4f}  {moe['stamp']}"
            )
        for i, (stamp, psnr, ssim, lpips) in enumerate(dyn_runs):
            tag = "dynamic v1" if i == 0 and len(dyn_runs) > 1 else (
                "dynamic v2" if i == len(dyn_runs) - 1 and len(dyn_runs) > 1 else "dynamic"
            )
            if len(dyn_runs) == 1:
                tag = "dynamic"
            print(f"{name:10s} {tag:28s} {psnr:7.3f} {ssim:7.4f} {lpips:7.4f}  {stamp}")
        if moe and dyn_runs:
            latest = dyn_runs[-1]
            print(
                f"{name:10s} {'Δ v2−MoE':28s} "
                f"{latest[1]-moe['metrics']['PSNR']:+7.3f} "
                f"{latest[2]-moe['metrics']['SSIM']:+7.4f} "
                f"{latest[3]-moe['metrics']['LPIPS']:+7.4f}"
            )
            if len(dyn_runs) > 1:
                old = dyn_runs[0]
                print(
                    f"{name:10s} {'Δ v2−v1':28s} "
                    f"{latest[1]-old[1]:+7.3f} "
                    f"{latest[2]-old[2]:+7.4f} "
                    f"{latest[3]-old[3]:+7.4f}"
                )
        print()


if __name__ == "__main__":
    main()
