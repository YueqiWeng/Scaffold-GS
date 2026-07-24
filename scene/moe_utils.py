#
# Helpers for MoE warm-start from Scaffold-GS baseline / flat-MoE checkpoints.
#

import os
import torch
from torch import nn

from scene.tree_moe import AttributeTreeMoE, DualRouter


def _expert0_state_dict(moe_state: dict) -> dict:
    prefix = "experts.0."
    return {k[len(prefix):]: v for k, v in moe_state.items() if k.startswith(prefix)}


def _collect_expert_state_dicts(state_dict: dict):
    """Extract experts.i.* blocks from a flat AttributeMoE state_dict."""
    expert_sds = []
    i = 0
    while True:
        prefix = f"experts.{i}."
        sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            break
        expert_sds.append(sd)
        i += 1
    return expert_sds


def _iter_leaf_experts(moe):
    if isinstance(moe, AttributeTreeMoE):
        return [leaf.expert for leaf in moe.leaf_nodes()]
    return list(moe.experts)


def _zero_routers(moe):
    if isinstance(moe, AttributeTreeMoE):
        for module in moe.modules():
            if isinstance(module, DualRouter):
                nn.init.zeros_(module.router.weight)
                nn.init.zeros_(module.router.bias)
                if module.gnn_router is not None:
                    nn.init.zeros_(module.gnn_router.weight)
                    nn.init.zeros_(module.gnn_router.bias)
        return
    if hasattr(moe, "router"):
        nn.init.zeros_(moe.router.weight)
        nn.init.zeros_(moe.router.bias)
    if getattr(moe, "gnn_router", None) is not None:
        nn.init.zeros_(moe.gnn_router.weight)
        nn.init.zeros_(moe.gnn_router.bias)


def _copy_flat_router_into_tree_root(moe: AttributeTreeMoE, state_dict: dict) -> bool:
    """
    When tree starts with E root leaves and checkpoint is flat MoE with E experts,
    copy the flat base router into the root DualRouter so warm-start matches MoE routing.
    GNN router stays zero; gamma left as initialized (typically small).
    """
    if not isinstance(moe, AttributeTreeMoE):
        return False
    root = moe.root
    if root is None or root.is_leaf or root.router is None:
        return False
    if len(root.child_nodes) != len(_collect_expert_state_dicts(state_dict)):
        return False
    # Flat AttributeMoE keys: router.weight / router.bias  (or base_router.*)
    for prefix in ("router.", "base_router."):
        w_key, b_key = f"{prefix}weight", f"{prefix}bias"
        if w_key in state_dict and b_key in state_dict:
            with torch.no_grad():
                root.router.router.weight.copy_(state_dict[w_key])
                root.router.router.bias.copy_(state_dict[b_key])
                if root.router.gnn_router is not None:
                    nn.init.zeros_(root.router.gnn_router.weight)
                    nn.init.zeros_(root.router.gnn_router.bias)
            print(
                f"[MoE] Copied flat router → tree root "
                f"({len(root.child_nodes)} leaves, prefix={prefix[:-1]})"
            )
            return True
    return False


def load_sequential_into_moe(moe, state_dict: dict):
    """
    Copy baseline Sequential or flat-MoE expert weights into leaf experts.

    Flat MoE with E experts → assign onto tree leaves (1:1 when counts match).
    If tree root has E children matching flat E, also copy flat router → root DualRouter.
    Otherwise routers are zero-initialized.
    """
    leaves = _iter_leaf_experts(moe)
    expert_sds = _collect_expert_state_dicts(state_dict)

    if expert_sds:
        for j, expert in enumerate(leaves):
            expert.load_state_dict(expert_sds[j % len(expert_sds)])
        if not _copy_flat_router_into_tree_root(moe, state_dict):
            _zero_routers(moe)
        return

    if len(leaves) <= 1:
        leaves[0].load_state_dict(state_dict)
        return

    for expert in leaves:
        expert.load_state_dict(state_dict)
    _zero_routers(moe)


def _load_jit_sequential(path: str) -> dict:
    return torch.jit.load(path, map_location="cuda").state_dict()


def load_baseline_checkpoints_into_model(gaussians, checkpoint_path: str):
    """
    Warm-start MoE / TreeMoE experts from Scaffold-GS or flat-MoE weights.

    Accepts:
      - checkpoints.pth (unite state_dict)
      - directory containing checkpoints.pth or opacity_mlp.pt / cov_mlp.pt / color_mlp.pt
    """
    if os.path.isdir(checkpoint_path):
        base = checkpoint_path
        unite = os.path.join(base, "checkpoints.pth")
        if os.path.isfile(unite):
            ckpt = torch.load(unite, map_location="cuda")
        else:
            ckpt = {
                "opacity_mlp": _load_jit_sequential(os.path.join(base, "opacity_mlp.pt")),
                "cov_mlp": _load_jit_sequential(os.path.join(base, "cov_mlp.pt")),
                "color_mlp": _load_jit_sequential(os.path.join(base, "color_mlp.pt")),
            }
    else:
        ckpt = torch.load(checkpoint_path, map_location="cuda")

    for key, moe in (
        ("opacity_mlp", gaussians.mlp_opacity),
        ("cov_mlp", gaussians.mlp_cov),
        ("color_mlp", gaussians.mlp_color),
    ):
        if key not in ckpt:
            raise KeyError(f"Missing {key} in {checkpoint_path}")
        load_sequential_into_moe(moe, ckpt[key])
    print(f"[MoE] Warm-started experts from: {checkpoint_path}")
