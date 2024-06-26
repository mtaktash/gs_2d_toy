import numpy as np
import torch
from torch import nn


def replace_tensor_to_optimizer(optimizer, tensor, name):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group["params"][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def cat_tensors_to_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1

        if group["name"] not in tensors_dict.keys():
            continue

        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)

        if stored_state is not None:

            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def prune_optimizer(optimizer, mask, param_names):
    optimizable_tensors = {}
    for group in optimizer.param_groups:

        if group["name"] not in param_names:
            continue

        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
