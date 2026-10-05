from __future__ import annotations

import copy

import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import create_dataloader
from ..registry import register_evaluator
from ..utils import TopOneAccuracy


class FeatureExtractor:
    def __init__(self, cfg, accelerator, model, tokenizer, dataset):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = accelerator.unwrap_model(model)
        self.tokenizer = tokenizer
        self.dataset = dataset

    @torch.no_grad()
    def text_classifier(self, dataset=None):
        dataset = dataset or self.dataset
        self.model.eval()

        classifier = list()
        for class_name in tqdm(dataset.class_name, desc="Text classifier", leave=False):
            if hasattr(dataset, "get_gpt_text"):
                text = dataset.get_gpt_text(class_name)
            else:
                text = [t.format(name=class_name) for t in dataset.prompt]
            tokens = self.tokenizer(text).to(self.device)

            with self.accelerator.autocast():
                feature = self.model.encode_text(tokens)

            feature = normalize(feature, dim=-1).mean(0)
            classifier.append(feature / feature.norm())

        return torch.stack(classifier, dim=0).to(self.device)

    @torch.no_grad()
    def image_features(self, dataset=None, desc="Query set"):
        dataset = dataset or self.dataset
        loader = create_dataloader(copy.deepcopy(dataset), self.cfg.eval.batch_size,
                                   self.cfg.eval.num_workers, is_train=False)
        self.model.eval()

        features, labels = list(), list()
        for data in tqdm(loader, total=len(loader), desc=desc, leave=False):
            x, y = data[0], data[1]
            x = x.to(self.device, memory_format=torch.channels_last)

            with self.accelerator.autocast():
                features.append(self.model.encode_image(x).detach().cpu())
            labels.append(y.detach().cpu())

        features = normalize(torch.cat(features, dim=0).to(self.device), dim=-1)
        return features, torch.cat(labels).to(self.device)


@register_evaluator("recalcon")
class Evaluator:
    def __init__(self, cfg, accelerator, model, tokenizer, dataset):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = accelerator.unwrap_model(model)
        self.dataset = dataset
        self.features = FeatureExtractor(cfg, accelerator, model, tokenizer, dataset)
        self.metric = TopOneAccuracy()

    @torch.no_grad()
    def __call__(self):
        self.metric.reset()
        self.model.eval()

        classifier = self.features.text_classifier()
        query, labels = self.features.image_features()

        logits = self.model.logit_scale.exp() * query @ classifier.mT
        if hasattr(self.dataset, "project_logits"):
            logits = self.dataset.project_logits(logits)

        if hasattr(self.dataset, "scoring"):
            preds = logits.argmax(dim=1, keepdim=True).view_as(labels)
            result = self.dataset.scoring(preds.cpu(), labels.cpu())[0]
            score = result.get("F1-macro_all", result.get("acc_worst_region"))
            return {"score": float(score) * 100}

        self.metric.update(logits, labels)
        return {"score": self.metric.compute() * 100}


APE_CRITERION_WEIGHT = {
    "flowers102": [0.4, 0.6], "ucf101": [0.3, 0.7], "oxfordiiitpet": [0.7, 0.3], "food101": [0.3, 0.7],
    "caltech101": [0.5, 0.5], "fgvc": [0.1, 0.9], "stanfordcars": [0.6, 0.4], "sun397": [0.7, 0.3],
    "eurosat": [0.9, 0.1], "imagenet": [0.7, 0.3], "dtd": [0.5, 0.5],
}

APE_INDICES = {
    "flowers102": 700, "ucf101": 700, "oxfordiiitpet": 800, "food101": 700, "caltech101": 800,
    "fgvc": 800, "stanfordcars": 500, "sun397": 800, "eurosat": 700, "imagenet": 500, "dtd": 400,
}

APE_SEARCH_SCALE = {
    "imagenet": (7, 7, 0.5), "flowers102": (7, 7, 1), "dtd": (7, 7, 1), "oxfordiiitpet": (7, 7, 1),
    "stanfordcars": (10, 7, 1), "ucf101": (10, 7, 1.5), "caltech101": (7, 7, 1), "food101": (7, 7, 1),
    "sun397": (10, 7, 1), "fgvc": (20, 10, 1), "eurosat": (30, 5, 1.5),
}


def pairwise_similarity(feats):
    cate_num, samp_num, feat_dim = feats.shape
    outer = torch.einsum("imf,jnf->ijf", feats, feats)
    mask = (1 - torch.eye(cate_num, device=feats.device)).unsqueeze(-1)
    return torch.mul(outer, mask).sum(dim=[0, 1]) / (cate_num * (cate_num - 1) * samp_num * samp_num)


def select_channels(text_classifier, sup_features, n_shot, dataset_name):
    w = APE_CRITERION_WEIGHT[dataset_name]
    n_class, feat_dim = text_classifier.shape
    feats = torch.cat([text_classifier.unsqueeze(1), sup_features.reshape(n_class, n_shot, feat_dim)], dim=1)
    criterion = -1 * w[0] * pairwise_similarity(feats) + w[1] * torch.var(text_classifier, dim=0)
    return torch.topk(criterion, k=APE_INDICES[dataset_name])[1]


@register_evaluator("recalcon_fewshot")
class FewShotEvaluator:
    def __init__(self, cfg, accelerator, model, tokenizer, dataset, dataset_name, train_dataset=None,
                 val_dataset=None):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = accelerator.unwrap_model(model)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset_name = dataset_name
        self.features = FeatureExtractor(cfg, accelerator, model, tokenizer, dataset)
        self.metric = TopOneAccuracy()

    @torch.no_grad()
    def support_set(self, model, repeats=10):
        loader = DataLoader(copy.deepcopy(self.train_dataset), self.cfg.eval.batch_size,
                            num_workers=self.cfg.eval.num_workers, drop_last=False, pin_memory=True, shuffle=False)
        loader.dataset.setup_prompt_transform()
        model.eval()

        features, labels = list(), list()
        for i in range(repeats):
            round_features = list()
            for data in tqdm(loader, total=len(loader), desc=f"Support set {i + 1}/{repeats}", leave=False):
                x, y = data[0], data[2]
                x = x.to(self.device, memory_format=torch.channels_last)
                with self.accelerator.autocast():
                    round_features.append(model.encode_image(x).detach().cpu())
                if i == 0:
                    labels.append(y.detach().cpu())
            features.append(torch.cat(round_features, dim=0))

        sup_features = torch.stack(features, dim=0).to(self.device).mean(0)
        return normalize(sup_features, dim=-1), torch.cat(labels).to(self.device)

    def cache_logits(self, query, sup_features, sup_labels, text_classifier, alpha, beta, gamma):
        similarity = query @ sup_features.mT
        key_logits = (sup_features @ text_classifier.mT).softmax(1)
        divergence = torch.sum(sup_labels * torch.log2((sup_labels + 1e-6) / (key_logits + 1e-6)),
                               dim=1).unsqueeze(-1)
        cache_values = sup_labels * (divergence * gamma).exp()
        return ((-1) * (beta - beta * similarity)).exp() @ cache_values.to(similarity.dtype) * alpha

    def search_hparams(self, base_logits, query, sup_features, sup_labels, text_classifier, labels):
        scale = APE_SEARCH_SCALE[self.dataset_name]
        steps = [200, 20, 20] if self.dataset_name != "imagenet" else [200, 20, 5]
        betas = [i * (scale[0] - 0.1) / steps[0] + 0.1 for i in range(steps[0])]
        alphas = [i * (scale[1] - 0.1) / steps[1] + 0.1 for i in range(steps[1])]
        gammas = [i * scale[2] / steps[2] for i in range(steps[2])]

        best = (1.0, -1.0, 0.5)
        best_accuracy = -1.0
        for beta in betas:
            for alpha in alphas:
                for gamma in gammas:
                    logits = base_logits + self.cache_logits(query, sup_features, sup_labels,
                                                             text_classifier, alpha, beta, gamma)
                    accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
                    if accuracy > best_accuracy:
                        best_accuracy, best = accuracy, (alpha, beta, gamma)
        return best

    @torch.no_grad()
    def zero_shot_logits(self, model, dataset, classifier):
        extractor = FeatureExtractor(self.cfg, self.accelerator, model, self.tokenizer, dataset)
        features, labels = extractor.image_features(dataset)
        return model.logit_scale.exp() * features @ classifier.mT, features, labels

    @torch.no_grad()
    def __call__(self):
        from ..models import build_clip

        self.metric.reset()
        clip_model = build_clip(self.cfg).to(self.device)
        clip_extractor = FeatureExtractor(self.cfg, self.accelerator, clip_model, self.tokenizer, self.dataset)

        clip_classifier = clip_extractor.text_classifier()
        query, labels = clip_extractor.image_features(self.dataset)
        val_features, val_labels = clip_extractor.image_features(self.val_dataset, desc="Val set")
        sup_features, sup_labels = self.support_set(clip_model)

        n_class = self.dataset.n_class
        n_shot = self.train_dataset.n_shot
        dim = clip_classifier.shape[1]
        sup_labels = torch.nn.functional.one_hot(sup_labels, n_class).view(n_class, -1, n_class).reshape(-1, n_class)
        sup_features = sup_features.view(n_class, n_shot, dim).reshape(-1, dim)

        channels = select_channels(clip_classifier, sup_features, n_shot, self.dataset_name)
        text_c = normalize(clip_classifier[:, channels], dim=-1)
        sup_c = normalize(sup_features[:, channels], dim=-1)
        query_c = normalize(query[:, channels], dim=-1)
        val_c = normalize(val_features[:, channels], dim=-1)

        alpha, beta, gamma = self.search_hparams(100.0 * val_features @ clip_classifier.mT, val_c, sup_c,
                                                 sup_labels, text_c, val_labels)
        logit_scale = clip_model.logit_scale.exp()
        clip_logits = logit_scale * query @ clip_classifier.mT + self.cache_logits(
            query_c, sup_c, sup_labels, text_c, alpha, beta, gamma)
        clip_val_logits = 100.0 * val_features @ clip_classifier.mT + self.cache_logits(
            val_c, sup_c, sup_labels, text_c, alpha, beta, gamma)

        classifier = self.features.text_classifier()
        ours_logits, _, _ = self.zero_shot_logits(self.model, self.dataset, classifier)
        ours_val_logits, _, _ = self.zero_shot_logits(self.model, self.val_dataset, classifier)

        best_alpha, best_accuracy = -1.0, -1.0
        for candidate in torch.arange(0, 10, 0.1):
            accuracy = (clip_val_logits + ours_val_logits * candidate).argmax(dim=-1).eq(val_labels).float().mean()
            if accuracy > best_accuracy:
                best_accuracy, best_alpha = accuracy, candidate

        self.metric.update(clip_logits + ours_logits * best_alpha, labels)
        return {"score": self.metric.compute() * 100}
