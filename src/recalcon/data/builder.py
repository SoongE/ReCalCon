from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from .augment import build_transform
from .datasets import (
    SUN397,
    UCF101,
    Caltech101,
    DescribableTextures,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageNet,
    ImageNetA,
    ImageNetR,
    ImageNetSketch,
    ImageNetV2,
    IWildCam,
    JsonDataset,
    ObjectNet,
    OxfordIIITPet,
    PCam,
    StanfordCars,
)
from .prompts import context_prompt, handcrafted_prompt


@dataclass(frozen=True)
class DatasetSpec:
    cls: type
    num_classes: int
    train: str | None = None
    valid: str | None = None
    test: str | None = None
    crop_pct: float | None = None


DATASETS = {
    "imagenet": DatasetSpec(ImageNet, 1000, "train", "val", "val"),
    "imagenet_r": DatasetSpec(ImageNetR, 200),
    "imagenet_a": DatasetSpec(ImageNetA, 200, crop_pct=0.875),
    "imagenet_v2": DatasetSpec(ImageNetV2, 1000),
    "imagenet_sketch": DatasetSpec(ImageNetSketch, 1000),
    "objectnet": DatasetSpec(ObjectNet, 113, crop_pct=0.875),
    "iwildcam": DatasetSpec(IWildCam, 182, "train", "val", "test"),
    "caltech101": DatasetSpec(Caltech101, 100, "train", "val", "test"),
    "flowers102": DatasetSpec(Flowers102, 102, "train", "val", "test"),
    "stanfordcars": DatasetSpec(StanfordCars, 196, "train", "val", "test"),
    "pcam": DatasetSpec(PCam, 2, "train", "val", "test"),
    "dtd": DatasetSpec(DescribableTextures, 47, "train", "val", "test"),
    "eurosat": DatasetSpec(EuroSAT, 10, "train", "val", "test"),
    "fgvc": DatasetSpec(FGVCAircraft, 100, "train", "val", "test"),
    "food101": DatasetSpec(Food101, 101, "train", "val", "test"),
    "oxfordiiitpet": DatasetSpec(OxfordIIITPet, 37, "train", "val", "test"),
    "sun397": DatasetSpec(SUN397, 397, "train", "val", "test"),
    "ucf101": DatasetSpec(UCF101, 101, "train", "val", "test"),
}

GROUPS = {
    "imagenet_ds": ["imagenet", "imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_sketch", "objectnet"],
    "imagenet_xd": ["imagenet", "caltech101", "eurosat", "fgvc", "flowers102", "food101",
                    "oxfordiiitpet", "stanfordcars", "sun397", "dtd", "ucf101"],
    "transfer": ["caltech101", "pcam", "imagenet", "flowers102", "stanfordcars", "iwildcam"],
}


def resolve_group(name):
    return list(GROUPS.get(name, [name]))


def get_spec(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. available: {sorted(DATASETS)} or groups {sorted(GROUPS)}")
    return DATASETS[name]


def build_prompts(cfg, name):
    if cfg.model.n_ctx > 0:
        return (context_prompt(name, cfg.model.n_ctx, augmented=False),
                context_prompt(name, cfg.model.n_ctx, augmented=True))
    return handcrafted_prompt(name, augmented=False), handcrafted_prompt(name, augmented=True)


def create_dataset(cfg, name, split=None, is_train=False, paired_views=False, n_shot=0):
    spec = get_spec(name)
    prompt, aug_prompt = build_prompts(cfg, name)

    kwargs = dict(
        transform=build_transform(cfg, is_train, crop_pct=spec.crop_pct),
        n_shot=n_shot,
        prompt=prompt,
        aug_prompt=aug_prompt,
        paired_views=paired_views,
        text_prompt=cfg.data.text_prompt,
        k_ops=cfg.data.k_ops,
    )

    if split is None:
        split = spec.train if is_train else spec.test

    if cfg.data.n_shot > 0:
        return JsonDataset(cfg.data.root, name, split, **kwargs)
    return spec.cls(cfg.data.root, split, dataset_name=name, **kwargs)


def create_dataloader(dataset, batch_size, num_workers, is_train):
    return DataLoader(dataset, batch_size, shuffle=is_train, num_workers=num_workers,
                      drop_last=False, pin_memory=True)
