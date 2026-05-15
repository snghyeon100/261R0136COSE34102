"""Low-rank task-matrix modules for UNLEARN."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankTaskLinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank task matrix."""

    def __init__(self, linear, rank, alpha, module_name):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LowRankTaskLinear can only wrap nn.Linear, got {type(linear)}")
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.module_name = module_name
        out_features, in_features = linear.weight.shape

        for param in self.linear.parameters():
            param.requires_grad = False

        self.task_A = nn.Parameter(torch.empty(out_features, self.rank, dtype=linear.weight.dtype))
        self.task_B = nn.Parameter(torch.zeros(self.rank, in_features, dtype=linear.weight.dtype))
        nn.init.normal_(self.task_A, mean=0.0, std=0.01)

    def task_matrix(self):
        return self.task_A @ self.task_B

    def forward(self, input):
        task_weight = self.alpha * self.task_matrix()
        weight = self.linear.weight + task_weight.to(self.linear.weight.dtype)
        return F.linear(input, weight, self.linear.bias)


def _split_parent_child(model, module_name):
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    return parent, child_name


def _layer_index_from_name(module_name):
    parts = module_name.split(".")
    for marker in ("layers", "h"):
        if marker in parts:
            idx = parts.index(marker) + 1
            if idx < len(parts) and parts[idx].isdigit():
                return int(parts[idx])
    return None


def _in_layer_range(module_name, start, end):
    if start is None and end is None:
        return True
    layer_idx = _layer_index_from_name(module_name)
    if layer_idx is None:
        return False
    if start is not None and layer_idx < int(start):
        return False
    if end is not None and layer_idx >= int(end):
        return False
    return True


def _matches_target(module_name, target_modules):
    return any(module_name.endswith(str(target)) for target in target_modules)


def freeze_original_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def wrap_task_matrices(model, target_modules, rank, alpha, target_layer_start=None, target_layer_end=None):
    """Wrap matching linear modules and return wrapped module names."""
    freeze_original_parameters(model)
    matches = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_target(name, target_modules):
            if _in_layer_range(name, target_layer_start, target_layer_end):
                matches.append(name)

    for name in matches:
        parent, child_name = _split_parent_child(model, name)
        original = getattr(parent, child_name)
        setattr(parent, child_name, LowRankTaskLinear(original, rank=rank, alpha=alpha, module_name=name))

    return matches


def iter_task_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, LowRankTaskLinear):
            yield name, module


def task_parameter_count(model):
    return sum(param.numel() for _, module in iter_task_modules(model) for param in (module.task_A, module.task_B))


def task_matrix_norms(model):
    norms = []
    for _, module in iter_task_modules(model):
        norms.append(module.task_matrix().detach().float().norm())
    if not norms:
        return None, None
    stacked = torch.stack(norms)
    return stacked.mean(), stacked.max()


def save_task_matrices(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for wrapped_name, module in iter_task_modules(model):
        base_name = module.module_name
        file_stem = base_name.replace(".", "__")
        torch.save(
            {
                "A": module.task_A.detach().cpu(),
                "B": module.task_B.detach().cpu(),
                "alpha": module.alpha,
                "rank": module.rank,
                "module_name": base_name,
            },
            output_dir / f"{file_stem}.pt",
        )
        metadata.append(
            {
                "wrapped_name": wrapped_name,
                "module_name": base_name,
                "file": f"{file_stem}.pt",
                "alpha": module.alpha,
                "rank": module.rank,
            }
        )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"modules": metadata}, f, indent=2)
    return metadata


def apply_task_matrices(model, task_matrix_dir, unlearn_scale=1.0, dtype=None):
    task_matrix_dir = Path(task_matrix_dir)
    with open(task_matrix_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    for entry in metadata["modules"]:
        module = model.get_submodule(entry["module_name"])
        state = torch.load(task_matrix_dir / entry["file"], map_location="cpu")
        alpha = float(state.get("alpha", entry.get("alpha", 1.0)))
        factor_dtype = dtype or module.weight.dtype
        task_matrix = (state["A"].to(factor_dtype) @ state["B"].to(factor_dtype)) * (alpha * float(unlearn_scale))
        module.weight.data.sub_(task_matrix.to(device=module.weight.device, dtype=module.weight.dtype))
    return len(metadata["modules"])


def assert_only_task_matrices_trainable(model):
    bad = []
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
            if not (name.endswith(".task_A") or name.endswith(".task_B")):
                bad.append(name)
    if bad:
        raise RuntimeError(f"Non-task parameters are trainable: {bad[:10]}")
    return trainable

