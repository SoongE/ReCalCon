from __future__ import annotations

import numpy as np
from timm.data import create_transform
from timm.data.auto_augment import RandAugment as TimmRandAugment
from torchvision.transforms import Compose, Normalize

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

RAND_AUG_WORDS = {
    "AutoContrast": "auto contrasted",
    "Equalize": "equalized",
    "Invert": "inverted",
    "Rotate": "rotated",
    "Posterize": "posterized",
    "Solarize": "solarized",
    "Color": "colored",
    "Contrast": "contrasted",
    "Brightness": "brighter",
    "BrightnessIncreasing": "more brighter",
    "Sharpness": "sharpen",
    "PosterizeIncreasing": "more posterized",
    "SolarizeAdd": "adding solarized",
    "SolarizeIncreasing": "increasing solarized",
    "ColorIncreasing": "color factor increased",
    "ContrastIncreasing": "contrasted",
    "SharpnessIncreasing": "more sharpen",
    "ShearX": "shear to x",
    "ShearY": "shear to y",
    "TranslateXRel": "translated by x",
    "TranslateYRel": "translated by y",
}

AUG_WORD_POOL = sorted(set(RAND_AUG_WORDS.values()))


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights
        self.replace = choice_weights is None

    def __call__(self, img):
        ops = np.random.choice(self.ops, self.num_layers, replace=self.replace, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img, ops


def build_transform(cfg, is_train, crop_pct=None):
    d = cfg.data
    return create_transform(
        (3, d.image_size, d.image_size),
        is_training=is_train,
        scale=tuple(d.scale),
        ratio=tuple(d.ratio),
        hflip=d.hflip,
        vflip=d.vflip,
        color_jitter=None,
        auto_augment=d.randaug if is_train else None,
        interpolation=d.interpolation,
        mean=CLIP_MEAN,
        std=CLIP_STD,
        crop_pct=d.crop_pct if crop_pct is None else crop_pct,
    )


def split_transform(transform, k_ops=None):
    ops = list(transform.transforms)
    norm_idx = next(i for i, t in enumerate(ops) if isinstance(t, Normalize))
    tensor_idx = norm_idx - 1

    ra_idx = next((i for i, t in enumerate(ops) if isinstance(t, TimmRandAugment)), None)
    if ra_idx is None:
        return Compose(ops[:tensor_idx]), None, Compose(ops[tensor_idx:])

    ra = ops[ra_idx]
    randaug = RandAugment(ra.ops, k_ops or ra.num_layers, ra.choice_weights)
    return Compose(ops[:ra_idx]), randaug, Compose(ops[ra_idx + 1:])


def describe_ops(ops):
    return " and ".join(RAND_AUG_WORDS[op.name] for op in ops)


def random_words(rng, k):
    return " and ".join(rng.sample(AUG_WORD_POOL, k))
