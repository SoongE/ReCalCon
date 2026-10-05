from __future__ import annotations

import torch
from torch import nn

from . import clip
from .registry import get_model_builder, register_model

_INIT_TEXT = [[320, 9388, 539, 320],
              [320, 9388, 539, 518],
              [320, 3610, 539, 320],
              [320, 3610, 539, 518],
              [320, 3086, 539, 320],
              [320, 3086, 539, 518],
              [320, 1125, 539, 320],
              [320, 1125, 539, 1346],
              [320, 1125, 539, 607],
              [320, 1125, 539, 637],
              [320, 1125, 539, 518],
              [320, 21339, 539, 320],
              [320, 21339, 539, 518],
              [320, 28891, 539, 320],
              [320, 28891, 539, 518],
              [320, 8757, 539, 320],
              [320, 8757, 539, 518],
              [320, 5269, 539, 320],
              [320, 5269, 539, 518],
              [320, 6325, 539, 320],
              [320, 6325, 539, 518]]

_ADAPTER_RATIO = {"mlp": 4.0, "small_mlp": 0.5}


def build_adapter(dim: int, kind: str) -> nn.Sequential:
    if kind not in _ADAPTER_RATIO:
        raise ValueError(f"unknown model.adapter '{kind}'; expected one of {sorted(_ADAPTER_RATIO)}")
    hidden = int(dim * _ADAPTER_RATIO[kind])
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim),
        nn.LayerNorm(dim),
    )


def fuse(x: torch.Tensor, lam) -> torch.Tensor:
    rolled = x.roll(-1, 0).mul_(1.0 - lam)
    rolled.add_(x.mul(lam))
    return torch.concat([x, rolled])


def encode_image(self, image, lam=1.0):
    x = self.visual(image.type(self.dtype))

    if self.training and lam != 1.0:
        x = fuse(x, lam)

    if self.vision_projection is not None:
        x = x @ self.vision_projection

    if self.vision_adapter is None:
        return x
    out = self.vision_adapter(x)
    return out + x if self.adapter_residual else out


def encode_text(self, text, lam=1.0):
    x = self.token_embedding(text).type(self.dtype)

    if self.learnable_context is not None:
        x[:, 1:1 + self.learnable_context.shape[0], :] = self.learnable_context

    x = x + self.positional_embedding.type(self.dtype)

    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x).type(self.dtype)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

    if self.training and lam != 1.0:
        x = fuse(x, lam)

    x = x @ self.text_projection

    if self.language_adapter is None:
        return x
    out = self.language_adapter(x)
    return out + x if self.adapter_residual else out


def forward_features(self, image, text, img_lam=1.0, txt_lam=1.0):
    image_features = self.encode_image(image, lam=img_lam)
    text_features = self.encode_text(text, lam=txt_lam)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return image_features, text_features


def _bind(model, fn, name):
    setattr(model, name, fn.__get__(model, model.__class__))


@register_model("recalcon")
def build_recalcon(cfg) -> nn.Module:
    m = cfg.model
    model, _ = clip.load(m.backbone)

    for param in model.parameters():
        param.requires_grad = not m.freeze

    embed_dim = model.text_projection.shape[1]
    text_width = model.token_embedding.embedding_dim

    model.adapter_residual = m.adapter_residual
    model.vision_projection = None
    model.vision_adapter = None
    model.language_adapter = None
    model.learnable_context = None

    if m.vision_adapter:
        if getattr(model.visual, "proj", None) is not None:
            model.vision_projection = model.visual.proj
            model.visual.proj = None
        model.vision_adapter = build_adapter(embed_dim, m.adapter)

    if m.language_adapter:
        model.language_adapter = build_adapter(embed_dim, m.adapter)

    if m.n_ctx > 0:
        if m.init_text:
            if m.n_ctx != len(_INIT_TEXT[0]):
                raise ValueError(f"model.init_text needs model.n_ctx == {len(_INIT_TEXT[0])}, got {m.n_ctx}")
            with torch.no_grad():
                init = model.token_embedding(torch.tensor(_INIT_TEXT)).type(model.dtype).mean(0)
        else:
            init = torch.empty(m.n_ctx, text_width, dtype=model.dtype).normal_(mean=0.0, std=0.02)
        model.learnable_context = nn.Parameter(init, requires_grad=True)

    _bind(model, encode_image, "encode_image")
    _bind(model, encode_text, "encode_text")
    _bind(model, forward_features, "forward")
    return model


@register_model("clip")
def build_clip(cfg) -> nn.Module:
    model, _ = clip.load(cfg.model.backbone)

    for param in model.parameters():
        param.requires_grad = False

    model.adapter_residual = False
    model.vision_projection = None
    model.vision_adapter = None
    model.language_adapter = None
    model.learnable_context = None

    _bind(model, encode_image, "encode_image")
    _bind(model, encode_text, "encode_text")
    _bind(model, forward_features, "forward")
    return model


def build_tokenizer():
    def tokenize(texts):
        return clip.tokenize(texts, truncate=True).long()
    return tokenize


def build_model(cfg):
    return get_model_builder(cfg.model.name)(cfg), build_tokenizer()
