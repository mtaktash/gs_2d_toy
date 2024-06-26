import math

import torch
from optimizer_params import (
    cat_tensors_to_optimizer,
    get_expon_lr_func,
    prune_optimizer,
    replace_tensor_to_optimizer,
    replace_tensors_to_optimizer,
)
from rendering import render_alpha_blend
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


def inverse_sigmoid(x, epsilon=1e-5):
    return torch.log((x + epsilon) / (1 - x + epsilon))


def op_sigmoid(x, k=100, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))


def rotation_matrix(theta):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # rotation matrix in 2D
    # [cos(theta), -sin(theta)]
    # [sin(theta), cos(theta)]

    R = torch.stack(
        [
            torch.concat([cos_theta, -sin_theta], dim=-1),
            torch.concat([sin_theta, cos_theta], dim=-1),
        ],
        dim=-2,
    )

    return R


def covariance_matrix(scale, rotation_matrix):
    # scale matrix
    S = torch.eye(scale.shape[1]).to(scale.device).unsqueeze(0) * scale.unsqueeze(-1)

    # rotation matrix
    R = rotation_matrix

    # covariance matrix
    # Σ = R S S^T R^T
    R_T = torch.transpose(R, -1, -2)
    S_T = torch.transpose(S, -1, -2)
    covariance = R @ S @ S_T @ R_T

    # add a small jitter to the covariance matrix to make it invertible
    # not having this causes NaNs in the gaussian pdf calculation randomly
    jitter = 1e-6 * torch.eye(covariance.shape[-1]).to(covariance.device).unsqueeze(0)
    covariance = covariance + jitter

    return covariance


def combined_loss(pred, target, lambda_param):
    ssim_loss = StructuralSimilarityIndexMeasure().to(pred.device)
    l1_loss = nn.L1Loss()

    d_ssim = (1 - ssim_loss(pred[None, None, ...], target[None, None, ...])) / 2
    return (1 - lambda_param) * l1_loss(pred, target) + lambda_param * d_ssim


N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n + 1):
        binoms[n, k] = math.comb(n, k)


class Gaussian2DImage(nn.Module):
    def __init__(self, num_gaussians, width, height, bg_color=[0.0, 0.0, 0.0]):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.width = width
        self.height = height
        self.bg_color = bg_color

        means = torch.rand(num_gaussians, 2)
        means = 2 * means - 1
        self.means = nn.Parameter(means, requires_grad=True)

        # scale of gaussian
        scales = torch.randn((num_gaussians, 2))
        scales = scales - math.exp(1)
        self.scales = nn.Parameter(scales, requires_grad=True)

        # rotation angle of gaussian
        thetas = torch.randn((num_gaussians, 1))
        self.thetas = nn.Parameter(thetas, requires_grad=True)

        # learnable opacities
        opacities = inverse_sigmoid(torch.ones((num_gaussians, 1)) * 0.1)
        self.opacities = nn.Parameter(opacities, requires_grad=True)

    def create_optimizer(self):
        params = [
            {"params": [self.means], "lr": 0.01, "name": "means"},
            {"params": [self.scales], "lr": 0.005, "name": "scales"},
            {"params": [self.thetas], "lr": 0.001, "name": "thetas"},
            {"params": [self.opacities], "lr": 0.05, "name": "opacities"},
        ]

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.mean_scheduler = get_expon_lr_func(0.01, 0.001, max_steps=30000)

    def update_learning_rate(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "means":
                lr = self.mean_scheduler(step)
                param_group["lr"] = lr
                return lr

    def get_means(self):
        return self.means

    def get_scales(self):
        # an exponential activation function for the scale of the covariance for smooth gradients
        return torch.exp(self.scales)

    def get_thetas(self):
        # theta might overflow and cause NaNs.
        # here is an activation function to limit theta from 0 to 2pi
        # this function activates between 0 and 2 pi
        return torch.sigmoid(self.thetas) * 2 * torch.pi

    def get_opacities(self):
        # sigmoid activation function for 𝛼 to constrain it in the [0 − 1) range and obtain smooth gradients
        return torch.sigmoid(self.opacities)

    def forward(self):
        means = self.get_means()
        scales = self.get_scales()
        thetas = self.get_thetas()
        opacities = self.get_opacities()

        covariances = covariance_matrix(scales, rotation_matrix(thetas))
        image = render_alpha_blend(self.width, self.height, means, covariances, opacities)
        return image

    def replace_params(self, optimizable_tensors):
        self.means = optimizable_tensors["means"]
        self.scales = optimizable_tensors["scales"]
        self.thetas = optimizable_tensors["thetas"]
        self.opacities = optimizable_tensors["opacities"]

    def densification_postfix(self, new_means, new_scales, new_thetas, new_opacities):
        tensors_dict = {
            "means": new_means,
            "scales": new_scales,
            "thetas": new_thetas,
            "opacities": new_opacities,
        }
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, tensors_dict)
        self.replace_params(optimizable_tensors)

    def reset_opacity(self, threshold=0.01):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacities(), torch.ones_like(self.opacities) * threshold),
        )
        optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacities")
        self.opacities = optimizable_tensors["opacities"]

    def densify_and_prune(self, max_grad=0.002, scale_threshold=6.0, min_opacity=0.005, max_screen_size=20):
        print("n_gaussians before:", self.means.shape[0])

        grad = self.means.grad.detach().clone()
        grad_norm = torch.norm(grad, dim=1)

        self.densify_and_clone(grad_norm, max_grad, scale_threshold)
        print("n_gaussians after clone:", self.means.shape[0])

        self.densify_and_split(grad_norm, max_grad, scale_threshold)
        print("n_gaussians after split:", self.means.shape[0])

        prune_mask = (self.get_opacities() < min_opacity).squeeze()

        if max_screen_size:
            extent = math.sqrt(self.width**2 + self.height**2)
            big_points_ws = torch.max(self.get_scales(), dim=1).values > 0.1 * extent
            big_points_vs = 2 * torch.norm(self.get_scales()[:, :2], dim=1) > max_screen_size

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        optimizable_tensors = prune_optimizer(
            self.optimizer,
            ~prune_mask,
            param_names=["means", "scales", "thetas", "opacities"],
        )
        self.replace_params(optimizable_tensors)
        print("n_gaussians after prune:", self.means.shape[0])

    def densify_and_clone(self, grad_norm, grad_threshold, scale_threshold):

        scale = self.get_scales()
        scale_norm = torch.norm(scale, dim=1)

        selected_pts_mask = grad_norm >= grad_threshold
        selected_pts_mask = selected_pts_mask & (scale_norm < scale_threshold)

        if selected_pts_mask.sum() == 0:
            return

        new_means = self.means[selected_pts_mask]
        new_scales = self.scales[selected_pts_mask]
        new_thetas = self.thetas[selected_pts_mask]
        new_opacities = self.opacities[selected_pts_mask]

        self.densification_postfix(new_means, new_scales, new_thetas, new_opacities)

    def densify_and_split(self, grad_norm, grad_threshold, scale_threshold):

        n_gaussians = self.means.shape[0]
        padded_grad = torch.zeros(n_gaussians, device=self.means.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm

        scale = self.get_scales()
        scale_norm = torch.norm(scale, dim=1)

        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = selected_pts_mask & (scale_norm >= scale_threshold)

        if selected_pts_mask.sum() == 0:
            return

        split_num = 2
        num_dims = self.means.shape[1]

        scales_sample = self.get_scales()[selected_pts_mask].repeat(split_num, 1)
        means_sample = torch.zeros((scales_sample.size(0), num_dims), device=scale.device)
        samples = torch.normal(mean=means_sample, std=scales_sample)

        # a workaround to work in both dummy 3d and 2d settings
        R = rotation_matrix(self.get_thetas()[selected_pts_mask]).repeat(split_num, 1, 1)

        old_means = self.get_means()[selected_pts_mask].repeat(split_num, 1)
        new_means = torch.bmm(R, samples.unsqueeze(-1)).squeeze(-1) + old_means

        new_scales = self.get_scales()[selected_pts_mask].repeat(split_num, 1) / 1.6
        new_scales = torch.log(new_scales)  # reverse the activation for scale

        new_thetas = self.thetas[selected_pts_mask].repeat(split_num, 1)
        new_opacities = self.opacities[selected_pts_mask].repeat(split_num, 1)

        self.densification_postfix(new_means, new_scales, new_thetas, new_opacities)

        prune_mask = torch.cat(
            (selected_pts_mask, torch.zeros(split_num * selected_pts_mask.sum(), device=self.means.device, dtype=bool))
        )
        optimizable_tensors = prune_optimizer(
            self.optimizer,
            ~prune_mask,
            param_names=["means", "scales", "thetas", "opacities"],
        )
        self.replace_params(optimizable_tensors)

    def _compute_relocation(self, opacities_old, scales_old, N):
        opacities_new = opacities_old / N
        N_expanded = N.repeat(1, scales_old.shape[1])
        scales_new = scales_old / torch.sqrt(N_expanded)
        return opacities_new, scales_new

    def _update_params(self, idxs, ratio):
        new_opacities, new_scales = self._compute_relocation(
            opacities_old=self.get_opacities()[idxs], scales_old=self.get_scales()[idxs], N=ratio[idxs] + 1
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacities = inverse_sigmoid(new_opacities)

        new_scales = torch.log(new_scales)

        return (
            self.means[idxs],
            new_scales,
            self.thetas[idxs],
            new_opacities,
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacities()[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self.means[dead_indices].data,
            self.scales[dead_indices].data,
            self.thetas[dead_indices].data,
            self.opacities[dead_indices].data,
        ) = self._update_params(reinit_idx, ratio)

        self.scales[reinit_idx].data = self.scales[dead_indices].data
        self.opacities[reinit_idx].data = self.opacities[dead_indices].data

        tensors_dict = {
            "means": self.means,
            "scales": self.scales,
            "thetas": self.thetas,
            "opacities": self.opacities,
        }

        optimizable_tensors = replace_tensors_to_optimizer(self.optimizer, tensors_dict, indices=reinit_idx)
        self.replace_params(optimizable_tensors)

    def add_new_gs(self, cap_max):
        current_num_points = self.opacities.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacities().squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_means,
            new_scales,
            new_thetas,
            new_opacities,
        ) = self._update_params(add_idx, ratio)

        self.scales[add_idx].data = new_scales
        self.opacities[add_idx].data = new_opacities

        self.densification_postfix(new_means, new_scales, new_thetas, new_opacities)

        tensors_dict = {
            "means": self.means,
            "scales": self.scales,
            "thetas": self.thetas,
            "opacities": self.opacities,
        }

        optimizable_tensors = replace_tensors_to_optimizer(self.optimizer, tensors_dict, indices=add_idx)
        self.replace_params(optimizable_tensors)
        return num_gs

    def add_means_noise(self, noise_lr, means_lr):
        actual_covariance = covariance_matrix(self.get_scales(), rotation_matrix(self.get_thetas()))
        noise = torch.randn_like(self.means) * (op_sigmoid(1 - self.get_opacities())) * noise_lr * means_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self.means.data.add_(noise)
