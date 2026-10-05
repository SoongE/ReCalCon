from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

MODEL_REGISTRY: dict[str, Callable] = {}
TRAINER_REGISTRY: dict[str, type] = {}
EVALUATOR_REGISTRY: dict[str, type] = {}


def _register(store: dict, kind: str, name: str):
    def deco(obj):
        if name in store:
            raise KeyError(f"{kind} '{name}' already registered")
        store[name] = obj
        return obj
    return deco


def register_model(name: str) -> Callable[[T], T]:
    return _register(MODEL_REGISTRY, "model", name)


def register_trainer(name: str) -> Callable[[type[T]], type[T]]:
    return _register(TRAINER_REGISTRY, "trainer", name)


def register_evaluator(name: str) -> Callable[[type[T]], type[T]]:
    return _register(EVALUATOR_REGISTRY, "evaluator", name)


def _get(store: dict, kind: str, name: str):
    if name not in store:
        raise KeyError(f"unknown {kind} '{name}'. registered: {sorted(store)}")
    return store[name]


def get_model_builder(name: str) -> Callable:
    return _get(MODEL_REGISTRY, "model", name)


def get_trainer_cls(name: str) -> type:
    return _get(TRAINER_REGISTRY, "trainer", name)


def get_evaluator_cls(name: str) -> type:
    return _get(EVALUATOR_REGISTRY, "evaluator", name)
