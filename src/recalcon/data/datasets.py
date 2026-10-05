from __future__ import annotations

import gzip
import json
import os
import pickle
import random
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torchvision.datasets import PCAM as TorchPCAM
from torchvision.datasets import SUN397 as TorchSUN397
from torchvision.datasets import EuroSAT as TorchEuroSAT
from torchvision.datasets import FGVCAircraft as TorchFGVCAircraft
from torchvision.datasets import Flowers102 as TorchFlowers102
from torchvision.datasets import Food101 as TorchFood101
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet as TorchImageNet
from torchvision.datasets import OxfordIIITPet as TorchOxfordIIITPet
from torchvision.datasets import StanfordCars as TorchStanfordCars

from .augment import describe_ops, random_words, split_transform
from .class_names import (
    CALTECH101_CLASS_NAME,
    DESCRIBABLE_TEXTURES_CLASS_NAME,
    EUROSAT_CLASS_NAME,
    FGVC_CLASS_NAME,
    FLOWERS102_CLASS_NAME,
    FOOD101_CLASS_NAME,
    IMAGENET_CLASS_NAME,
    IWILDCAM_CLASS_NAME,
    OXFORD_IIIT_PETS_CLASS_NAME,
    PCAM_CLASS_NAME,
    STANFORDCARS_CLASS_NAME,
    SUN397_CLASS_NAME,
    UCF101_CLASS_NAME,
)

IMAGENET_A_CLASS_NUMBER = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108,
    110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386,
    397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483,
    486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606,
    607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
    763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845,
    847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954,
    956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988,
]

IMAGENET_R_CLASS_NUMBER = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113,
    122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207,
    208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
    291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344,
    347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437,
    441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594,
    596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820,
    824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951,
    953, 954, 957, 963, 965, 967, 980, 981, 983, 988,
]


class VLMDataset(Dataset):
    dataset_path = ""
    n_class = 0

    def __init__(self, root, imgs, targets, class_names, transform=None, target_transform=None, n_shot=0,
                 prompt=None, aug_prompt=None, paired_views=False, text_prompt="align", k_ops=2,
                 dataset_name=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_name = dataset_name or self.__class__.__name__.lower()

        self._prompt = list(prompt) if prompt else ["a photo of a {name}."]
        self._aug_prompt = list(aug_prompt) if aug_prompt else ["a {aug} photo of a {name}."]
        self.paired_views = paired_views
        self.text_prompt = text_prompt
        self.k_ops = k_ops

        self.randaug = None
        self.pre_processing = None
        self.post_processing = None
        self._rng = random.Random()

        self.set_class_name(class_names)
        self.origin_imgs, self.origin_targets = imgs, targets
        self.sampling(n_shot)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def prompt(self):
        return self._prompt

    def set_class_name(self, class_names):
        self.class_name = class_names
        self._num2name = {i: name for i, name in enumerate(class_names)}
        self._name2num = {name: i for i, name in enumerate(class_names)}

    def num2str(self, num):
        return self._num2name.get(num, f"{num} does not exist in {self.dataset_path}")

    def str2num(self, class_name):
        return self._name2num.get(class_name, f"{class_name} does not exist in {self.dataset_path}")

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.origin_imgs)):
            data_dict[self.origin_targets[i]].append(self.origin_imgs[i])
        return data_dict

    def sampling(self, n_shot):
        self.n_shot = n_shot

        if n_shot == 0:
            self.imgs, self.targets = self.origin_imgs, self.origin_targets
            return

        sample_dir = os.path.join(self.root, "sampling")
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, f"soonge_{self.dataset_name}_{n_shot}s.pkl")

        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                data = pickle.load(f)
            self.imgs = [os.path.join(self.root, x) for x in data["s_imgs"]]
            self.targets = data["s_targets"]
            return

        s_imgs, s_targets = list(), list()
        for class_num, items in self._data_dict().items():
            if len(items) >= n_shot:
                s_imgs.extend(random.sample(items, n_shot))
            else:
                s_imgs.extend(random.choices(items, k=n_shot))
            s_targets.extend([class_num for _ in range(n_shot)])

        with open(sample_path, "wb") as f:
            pickle.dump(dict(s_imgs=s_imgs, s_targets=s_targets), f, protocol=3)

        self.imgs, self.targets = s_imgs, s_targets

    def setup_prompt_transform(self):
        self.pre_processing, self.randaug, self.post_processing = split_transform(self.transform, self.k_ops)
        if self.text_prompt == "align" and self.randaug is None:
            raise ValueError("data.text_prompt='align' needs data.randaug; without image ops there is nothing "
                             "for the prompt to describe (use 'fix' or 'misalign')")

    def _format(self, templates, target, aug=None):
        return self._rng.choice(templates).format(name=self.num2str(target), aug=aug)

    def clean_prompt(self, target):
        if self.text_prompt == "fix":
            return self._format(self._prompt, target)
        return self._format(self._aug_prompt, target, aug="original")

    def augmented_prompt(self, target, ops):
        if self.text_prompt == "fix":
            return self._format(self._prompt, target)
        if self.text_prompt == "misalign":
            return self._format(self._aug_prompt, target, aug=random_words(self._rng, self.k_ops))
        return self._format(self._aug_prompt, target, aug=describe_ops(ops))

    @staticmethod
    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        img = self.loader(path)

        if not self.paired_views:
            return self.transform(img), target, self._format(self._prompt, target)

        base = self.pre_processing(img)
        if self.randaug is None:
            augmented, ops = self.pre_processing(img), ()
        else:
            augmented, ops = self.randaug(base)

        return (self.post_processing(base), self.post_processing(augmented), target,
                self.clean_prompt(target), self.augmented_prompt(target, ops))

    def __str__(self):
        return f"{self.__class__.__name__} | # class: {self.n_class} | root: {self.dataset_path}"

    @staticmethod
    def _split_warning(dataset_name, split, state):
        if split is not state:
            print(f'{colored("[DATASET_WARNING]", "red")} {dataset_name} does not support the "split" argument.')


class ImageNet(VLMDataset):
    dataset_path = "imageNet"
    n_class = 1000

    def __init__(self, root, split="val", **kwargs):
        dataset = TorchImageNet(os.path.join(root, self.dataset_path), split)
        super().__init__(root, [x[0] for x in dataset.imgs], dataset.targets, IMAGENET_CLASS_NAME, **kwargs)

    def _data_dict(self):
        train_dataset = TorchImageNet(os.path.join(self.root, self.dataset_path), "train")
        data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])
        return data_dict


class ImageNetX(VLMDataset):
    dataset_path = "imageNet-X"
    n_class = 1000
    class_number = range(0, 1000)

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        super().__init__(root, [x[0] for x in dataset.imgs], dataset.targets, IMAGENET_CLASS_NAME, **kwargs)

    def project_logits(self, logits):
        if logits.shape[-1] == self.n_class:
            return logits
        return logits[:, self.class_number]


class ImageNetR(ImageNetX):
    dataset_path = "imageNet-R"
    n_class = 200
    class_number = IMAGENET_R_CLASS_NUMBER


class ImageNetA(ImageNetX):
    dataset_path = "imageNet-A"
    n_class = 200
    class_number = IMAGENET_A_CLASS_NUMBER


class ImageNetSketch(ImageNetX):
    dataset_path = "imageNet-Sketch"


class ImageNetV2(VLMDataset):
    dataset_path = "imageNet-V2"
    n_class = 1000

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        imgs, targets = list(), list()
        for sample in glob(os.path.join(root, self.dataset_path, "*/*")):
            imgs.append(sample)
            targets.append(int(os.path.basename(os.path.dirname(sample))))
        super().__init__(root, imgs, targets, IMAGENET_CLASS_NAME, **kwargs)


class ObjectNet(VLMDataset):
    dataset_path = "objectnet-1.0"
    n_class = 113

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        self.folders_to_ids, self.classname_map = self._metadata(root)

        folders = sorted(self.folders_to_ids.keys())
        self.rev_class_idx_map, self.class_idx_map = dict(), dict()
        for idx, folder in enumerate(folders):
            self.rev_class_idx_map[idx] = self.folders_to_ids[folder]
            for imagenet_idx in self.rev_class_idx_map[idx]:
                self.class_idx_map[imagenet_idx] = idx

        imgs, targets = list(), list()
        for idx, folder in enumerate(folders):
            class_img = glob(os.path.join(root, self.dataset_path, "images", folder, "*"))
            imgs.extend(class_img)
            targets.extend([idx for _ in range(len(class_img))])

        super().__init__(root, imgs, targets, IMAGENET_CLASS_NAME, **kwargs)

    def _metadata(self, root):
        metadata = Path(os.path.join(root, self.dataset_path, "mappings"))

        with open(metadata / "folder_to_objectnet_label.json", "r") as f:
            folder_map = {v: k for k, v in json.load(f).items()}
        with open(metadata / "objectnet_to_imagenet_1k.json", "r") as f:
            objectnet_map = json.load(f)
        with open(metadata / "pytorch_to_imagenet_2012_id.json", "r") as f:
            pytorch_map = {v: k for k, v in json.load(f).items()}
        with open(metadata / "imagenet_to_label_2012_v2", "r") as f:
            imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

        folder_to_ids = dict()
        for objectnet_name, imagenet_names in objectnet_map.items():
            imagenet_ids = [int(imagenet_map[n]) for n in imagenet_names.split("; ")]
            folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

        return folder_to_ids, {v: k for k, v in folder_map.items()}

    def project_logits(self, logits):
        if logits.shape[1] == self.n_class:
            return logits
        logits = logits.detach() if torch.is_tensor(logits) else logits
        projected = torch.zeros((logits.shape[0], self.n_class), device=logits.device)
        for k, v in self.rev_class_idx_map.items():
            projected[:, k] = torch.amax(logits[:, v], dim=1)
        return projected

    @staticmethod
    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            width, height = img.size
            return img.crop((2, 2, width - 2, height - 2)).convert("RGB")


class IWildCam(VLMDataset):
    dataset_path = "iwildcam_v2.0"
    n_class = 182

    def __init__(self, root, split="test", **kwargs):
        import wilds

        assert split in ["train", "val", "test", "id_val", "id_test"]
        self.dataset = wilds.get_dataset(dataset="iwildcam", root_dir=root)

        split_idx = np.where(self.dataset.split_array == self.dataset.split_dict[split])[0]
        imgs = self.dataset._input_array[split_idx]
        for i in range(len(imgs)):
            imgs[i] = os.path.join(root, self.dataset_path, "train", imgs[i])
        targets = self.dataset.y_array[split_idx].numpy()
        self.meta_data = self.dataset.metadata_array

        super().__init__(root, imgs, targets, IWILDCAM_CLASS_NAME, **kwargs)

    def scoring(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, self.meta_data)


class Caltech101(VLMDataset):
    dataset_path = "caltech101"
    n_class = 100

    def __init__(self, root, split="test", **kwargs):
        assert split in ["train", "val", "test"]
        dataset = ImageFolder(os.path.join(root, self.dataset_path, "split", split))
        class_names = list(CALTECH101_CLASS_NAME)
        class_names[0] = "face"
        super().__init__(root, [x[0] for x in dataset.imgs], dataset.targets, class_names, **kwargs)


class Flowers102(VLMDataset):
    dataset_path = "flowers-102"
    n_class = 102

    def __init__(self, root, split="test", **kwargs):
        if split == "trainval":
            dataset = TorchFlowers102(root, "train")
            val_dataset = TorchFlowers102(root, "val")
            dataset._image_files.extend(val_dataset._image_files)
            dataset._labels.extend(val_dataset._labels)
        else:
            dataset = TorchFlowers102(root, split)
        super().__init__(root, dataset._image_files, dataset._labels, FLOWERS102_CLASS_NAME, **kwargs)

    def _data_dict(self):
        train_dataset = TorchFlowers102(self.root, "train")
        val_dataset = TorchFlowers102(self.root, "val")
        train_dataset._image_files.extend(val_dataset._image_files)
        train_dataset._labels.extend(val_dataset._labels)

        data_dict = defaultdict(list)
        for i in range(len(train_dataset)):
            data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return data_dict


class StanfordCars(VLMDataset):
    dataset_path = "stanford_cars"
    n_class = 196

    def __init__(self, root, split="test", **kwargs):
        dataset = TorchStanfordCars(root, split)
        imgs = [s[0] for s in dataset._samples]
        targets = [s[1] for s in dataset._samples]
        super().__init__(root, imgs, targets, STANFORDCARS_CLASS_NAME, **kwargs)


class PCam(VLMDataset):
    dataset_path = "pcam"
    n_class = 2

    def __init__(self, root, split="test", **kwargs):
        imgs, targets = self._load(root, split)
        super().__init__(root, imgs, targets, PCAM_CLASS_NAME, **kwargs)

    @staticmethod
    def _load(root, split):
        dataset = TorchPCAM(root, split)
        files = dataset._FILES[dataset._split]
        imgs = dataset.h5py.File(dataset._base_folder / files["images"][0])["x"][:]
        targets = dataset.h5py.File(dataset._base_folder / files["targets"][0])["y"][:, 0, 0, 0]
        return imgs, targets

    def _data_dict(self):
        imgs, targets = self._load(self.root, "train")
        data_dict = defaultdict(list)
        for i in range(len(imgs)):
            data_dict[targets[i]].append(imgs[i])
        return data_dict

    @staticmethod
    def loader(data):
        return Image.fromarray(data).convert("RGB")


class DescribableTextures(VLMDataset):
    dataset_path = "dtd"
    n_class = 47

    def __init__(self, root, split="val", **kwargs):
        data_dir = Path(root) / self.dataset_path
        with open(data_dir / "labels" / f"{split}1.txt") as f:
            files = [line.strip() for line in f if line.strip()]
        class_to_idx = {c: i for i, c in enumerate(sorted({x.split("/")[0] for x in files}))}
        imgs = [str(data_dir / "images" / x) for x in files]
        targets = [class_to_idx[x.split("/")[0]] for x in files]
        super().__init__(root, imgs, targets, DESCRIBABLE_TEXTURES_CLASS_NAME, **kwargs)


class EuroSAT(VLMDataset):
    dataset_path = "eurosat"
    n_class = 10

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = TorchEuroSAT(root)
        super().__init__(root, [x[0] for x in dataset.imgs], dataset.targets, EUROSAT_CLASS_NAME, **kwargs)


class FGVCAircraft(VLMDataset):
    dataset_path = "fgvc-aircraft-2013b"
    n_class = 100

    def __init__(self, root, split="val", **kwargs):
        dataset = TorchFGVCAircraft(root, split)
        super().__init__(root, dataset._image_files, dataset._labels, FGVC_CLASS_NAME, **kwargs)

    def _data_dict(self):
        train_dataset = TorchFGVCAircraft(self.root, "train")
        data_dict = defaultdict(list)
        for i in range(len(train_dataset._image_files)):
            data_dict[train_dataset._labels[i]].append(train_dataset._image_files[i])
        return data_dict


class Food101(VLMDataset):
    dataset_path = "food-101"
    n_class = 101

    def __init__(self, root, split="test", **kwargs):
        dataset = TorchFood101(root, split)
        super().__init__(root, dataset._image_files, dataset._labels, FOOD101_CLASS_NAME, **kwargs)

    def _data_dict(self):
        train_dataset = TorchFood101(self.root, "train")
        data_dict = defaultdict(list)
        for i in range(len(train_dataset._image_files)):
            data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return data_dict


class OxfordIIITPet(VLMDataset):
    dataset_path = "oxford-iiit-pet"
    n_class = 37

    def __init__(self, root, split="test", **kwargs):
        dataset = TorchOxfordIIITPet(root, split)
        super().__init__(root, dataset._images, dataset._labels, OXFORD_IIIT_PETS_CLASS_NAME, **kwargs)

    def _data_dict(self):
        train_dataset = TorchOxfordIIITPet(self.root, "trainval")
        data_dict = defaultdict(list)
        for i in range(len(train_dataset._images)):
            data_dict[train_dataset._labels[i]].append(str(train_dataset._images[i]))
        return data_dict


class SUN397(VLMDataset):
    dataset_path = "SUN397"
    n_class = 397

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        self.dataset = TorchSUN397(root)
        super().__init__(root, self.dataset._image_files, self.dataset._labels, SUN397_CLASS_NAME, **kwargs)

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.dataset._image_files)):
            data_dict[self.dataset._labels[i]].append(str(self.dataset._image_files[i]))
        return data_dict


class UCF101(VLMDataset):
    dataset_path = "UCF-101-midframes"
    n_class = 101

    def __init__(self, root, split=None, **kwargs):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        super().__init__(root, [x[0] for x in dataset.imgs], dataset.targets, UCF101_CLASS_NAME, **kwargs)


GPT3_PROMPT_DIR = Path(__file__).resolve().parent / "gpt3_prompts"

JSON_SPLITS = {
    "imagenet": ("imageNet", "splits/split_soonge_imagenet.json.gz", "CuPL_prompts_imagenet.json", 1000),
    "caltech101": ("caltech101/101_ObjectCategories", "splits/split_zhou_Caltech101.json.gz",
                   "CuPL_prompts_caltech101.json", 100),
    "flowers102": ("flowers-102/jpg", "splits/split_zhou_OxfordFlowers.json.gz",
                   "CuPL_prompts_flowers102.json", 102),
    "food101": ("food-101/images", "splits/split_zhou_Food101.json.gz", "CuPL_prompts_food101.json", 101),
    "dtd": ("dtd/images", "splits/split_zhou_DescribableTextures.json.gz", "CuPL_prompts_dtd.json", 47),
    "oxfordiiitpet": ("oxford-iiit-pet/images", "splits/split_zhou_OxfordPets.json.gz",
                      "CuPL_prompts_oxfordpets.json", 37),
    "sun397": ("SUN397", "splits/split_zhou_SUN397.json.gz", "CuPL_prompts_sun397.json", 397),
    "ucf101": ("UCF-101-midframes", "splits/split_zhou_UCF101.json.gz", "CuPL_prompts_ucf101.json", 101),
    "stanfordcars": ("stanford_cars", "splits/split_zhou_StanfordCars.json.gz",
                     "CuPL_prompts_stanfordcars.json", 196),
    "eurosat": ("eurosat/2750", "splits/split_zhou_EuroSAT.json.gz", "CuPL_prompts_eurosat.json", 10),
    "fgvc": ("fgvc-aircraft-2013b", "splits/split_soonge_fgvc.json.gz", "CuPL_prompts_fgvcaircraft.json", 100),
}

JSON_CLASS_NAMES = {
    "imagenet": IMAGENET_CLASS_NAME,
    "caltech101": CALTECH101_CLASS_NAME,
    "flowers102": FLOWERS102_CLASS_NAME,
    "food101": FOOD101_CLASS_NAME,
    "dtd": DESCRIBABLE_TEXTURES_CLASS_NAME,
    "oxfordiiitpet": OXFORD_IIIT_PETS_CLASS_NAME,
    "sun397": SUN397_CLASS_NAME,
    "ucf101": UCF101_CLASS_NAME,
    "stanfordcars": STANFORDCARS_CLASS_NAME,
    "eurosat": EUROSAT_CLASS_NAME,
    "fgvc": FGVC_CLASS_NAME,
}


class JsonDataset(VLMDataset):
    def __init__(self, root, name, split, **kwargs):
        self.dataset_path, json_path, gpt3_path, self.n_class = JSON_SPLITS[name]

        path = os.path.join(root, json_path)
        with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
            samples = json.load(f)[split]
        imgs = [os.path.join(root, self.dataset_path, i) for i, _, _ in samples]
        targets = [t for _, t, _ in samples]

        with open(GPT3_PROMPT_DIR / gpt3_path) as f:
            self.gpt3_prompts = json.load(f)

        super().__init__(root, imgs, targets, JSON_CLASS_NAMES[name], dataset_name=name, **kwargs)

    def get_gpt_text(self, class_name):
        return [t.format(name=class_name) for t in self._prompt] + self.gpt3_prompts[class_name]

    def clean_prompt(self, target):
        return super().clean_prompt(target) + " " + self._rng.choice(self.gpt3_prompts[self.class_name[target]])

    def augmented_prompt(self, target, ops):
        return super().augmented_prompt(target, ops) + " " + self._rng.choice(
            self.gpt3_prompts[self.class_name[target]])
