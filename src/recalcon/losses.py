from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

_EPS = 1e-5


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def gather_all_features(*features: torch.Tensor) -> list[torch.Tensor]:
    return [_AllGather.apply(f) for f in features]


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = 0
        self.world_size = 1

    def row_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, image_features, text_features, labels, logit_scale, lam=1.0):
        if self.world_size > 1:
            image_features, text_features, labels = gather_all_features(image_features, text_features, labels)

        logits_per_image = logit_scale * torch.mm(image_features, text_features.t())
        targets = self.positive_mask(labels)

        if image_features.shape[0] != text_features.shape[0]:
            targets = self.fusion_targets(targets, lam)
            if image_features.shape[0] < text_features.shape[0]:
                targets = targets.t()

        return (self.row_loss(logits_per_image, targets)
                + self.row_loss(logits_per_image.t(), targets.t())) / 2

    @staticmethod
    def positive_mask(labels: torch.Tensor) -> torch.Tensor:
        return (labels.unsqueeze(-1) == labels.unsqueeze(0)).float()

    def fusion_targets(self, targets: torch.Tensor, lam) -> torch.Tensor:
        rolled = targets.clone()
        chunk = targets.size(0) // self.world_size
        for i in range(self.world_size):
            cols = slice(i * chunk, (i + 1) * chunk)
            rolled[:, cols] = rolled[:, cols].roll((0, -1), (0, 1))
        fused = targets * lam + rolled * (1.0 - lam)

        out = torch.zeros(targets.shape[0] * 2, targets.shape[1],
                          device=targets.device, dtype=targets.dtype)
        for i in range(self.world_size):
            src = slice(i * chunk, (i + 1) * chunk)
            out[i * 2 * chunk: i * 2 * chunk + chunk] = targets[src]
            out[i * 2 * chunk + chunk: (i + 1) * 2 * chunk] = fused[src]
        return out


class ReCalibratedContrastiveLoss(ContrastiveLoss):
    def row_loss(self, logits, targets):
        cardinality = targets.sum(dim=1)
        exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0]) + _EPS
        psi = (torch.sum(exp_logits * targets, dim=1, keepdim=True)
               / torch.sum(exp_logits * (1 - targets), dim=1, keepdim=True))
        log_prob = torch.log(F.softmax(logits, dim=-1) * psi + _EPS)
        return (torch.sum(-targets * log_prob, dim=-1) / cardinality).mean()


class SoftCrossEntropyLoss(ContrastiveLoss):
    def row_loss(self, logits, targets):
        cardinality = targets.sum(dim=1)
        loss = torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1) / cardinality
        return loss.mean()


class InfoNCELoss(ContrastiveLoss):
    def row_loss(self, logits, targets):
        cardinality = targets.sum(dim=1)
        logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0]) + _EPS
        loss = torch.sum(-targets * F.log_softmax(logits / cardinality.unsqueeze(1), dim=-1), dim=-1)
        return loss.mean()


class FocalContrastiveLoss(ContrastiveLoss):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def row_loss(self, logits, targets):
        cardinality = targets.sum(dim=1)
        log_prob = F.log_softmax(logits, dim=-1)
        focal = (1.0 - log_prob.exp()).clamp(min=0.0).pow(self.gamma)
        loss = torch.sum(-targets * focal * log_prob, dim=-1) / cardinality
        return loss.mean()


CRITERIONS = {
    "recalibrated": ReCalibratedContrastiveLoss,
    "softce": SoftCrossEntropyLoss,
    "infonce": InfoNCELoss,
    "focal": FocalContrastiveLoss,
}


def build_criterion(cfg) -> ContrastiveLoss:
    name = cfg.loss.name
    if name not in CRITERIONS:
        raise ValueError(f"unknown loss.name '{name}'; expected one of {sorted(CRITERIONS)}")
    if name == "focal":
        return FocalContrastiveLoss(gamma=cfg.loss.focal_gamma)
    return CRITERIONS[name]()
