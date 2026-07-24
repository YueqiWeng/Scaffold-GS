#!/usr/bin/env python3
"""
Re-run render + metrics for trained dynamic TreeMoE runs that failed at eval
(root-no-GNN checkpoint vs topology rebuild mismatch).
"""
from __future__ import annotations

import os
import sys
import glob
import logging
from argparse import ArgumentParser, Namespace

# Repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

# Prefer free GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from train import create_gaussian_model, render_sets, evaluate


def setup_logger(model_path: str) -> logging.Logger:
    logger = logging.getLogger(f"reeval:{model_path}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger


def reeval_one(model_path: str) -> None:
    model_path = os.path.abspath(model_path)
    ckpt = os.path.join(model_path, "point_cloud", "iteration_30000", "checkpoints.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    # Build argv so get_combined_args merges cfg_args from model_path
    sys.argv = ["reeval_dynamic.py", "-m", model_path, "--eval"]
    parser = ArgumentParser(description="Re-eval dynamic TreeMoE")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    args = get_combined_args(parser)

    # Ensure tree MoE flags survive cfg merge
    if not getattr(args, "moe_arch", None):
        args.moe_arch = "dynamic_tree"
    args.model_path = model_path
    args.eval = True

    logger = setup_logger(model_path)
    logger.info(f"Re-eval {model_path}")
    logger.info(
        f"moe_arch={args.moe_arch} branch={getattr(args,'moe_branch_factor',None)} "
        f"use_gnn={getattr(args,'use_gnn',None)}"
    )

    dataset = lp.extract(args)
    dataset.model_path = model_path
    pipe = pp.extract(args)

    visible_count = render_sets(
        dataset, args.iteration, pipe, skip_train=True, skip_test=False, logger=logger
    )
    evaluate(model_path, visible_count=visible_count, logger=logger)
    results = os.path.join(model_path, "results.json")
    if os.path.isfile(results):
        logger.info(f"Wrote {results}")
    else:
        raise RuntimeError(f"No results.json after eval: {model_path}")


def main():
    runs = [
        "outputs/mipnerf360/stump/dynamic_tree_moe_gnn/2026-07-20_23:26:44",
        "outputs/mipnerf360/garden/dynamic_tree_moe_gnn/2026-07-21_00:24:57",
        "outputs/mipnerf360/kitchen/dynamic_tree_moe_gnn/2026-07-21_01:44:39",
        "outputs/mipnerf360/bicycle/dynamic_tree_moe_gnn/2026-07-21_03:19:29",
        "outputs/mipnerf360/bonsai/dynamic_tree_moe_gnn/2026-07-21_05:00:07",
        "outputs/mipnerf360/counter/dynamic_tree_moe_gnn/2026-07-21_06:38:13",
        "outputs/mipnerf360/room/dynamic_tree_moe_gnn/2026-07-21_07:49:23",
    ]
    # Allow override: python scripts/reeval_fullwin2.py path1 path2 ...
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        runs = sys.argv[1:]

    for run in runs:
        print("=" * 72)
        reeval_one(run)
    print("ALL REEVAL DONE")


if __name__ == "__main__":
    main()
