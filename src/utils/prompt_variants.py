"""
Utilities for generating prompt variants used by the residual comparison experiment.

Variants can be registered dynamically and are referenced by name inside configs.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional

VariantFn = Callable[[str, Dict[str, object]], str]


class PromptVariantRegistry:
    """Registry for prompt transformation functions."""

    def __init__(self) -> None:
        self._registry: Dict[str, VariantFn] = {}

    def register(self, name: str, fn: VariantFn) -> VariantFn:
        if name in self._registry:
            raise ValueError(f"Prompt variant '{name}' is already registered.")
        self._registry[name] = fn
        return fn

    def get(self, name: str) -> VariantFn:
        if name not in self._registry:
            raise KeyError(f"Prompt variant '{name}' is not registered.")
        return self._registry[name]

    def list_variants(self) -> List[str]:
        return sorted(self._registry.keys())

    def apply(self, name: str, prompt: str, options: Optional[Dict[str, object]] = None) -> str:
        fn = self.get(name)
        return fn(prompt, options or {})


REGISTRY = PromptVariantRegistry()


def register_prompt_variant(name: str) -> Callable[[VariantFn], VariantFn]:
    """Decorator for registering prompt variants."""

    def decorator(fn: VariantFn) -> VariantFn:
        return REGISTRY.register(name, fn)

    return decorator


def split_into_sentences(prompt: str) -> List[str]:
    """
    Lightweight sentence splitter (punctuation aware but not perfect). Suitable for
    rapid experimentation without heavy NLP dependencies.
    """
    sentences: List[str] = []
    current = []
    for char in prompt:
        current.append(char)
        if char in ".?!\n":
            sentences.append("".join(current).strip())
            current = []
    if current:
        sentences.append("".join(current).strip())
    return [s for s in sentences if s]


@register_prompt_variant("identity")
def variant_identity(prompt: str, _: Dict[str, object]) -> str:
    return prompt


@register_prompt_variant("mirror_halves")
def variant_mirror_halves(prompt: str, _: Dict[str, object]) -> str:
    midpoint = len(prompt) // 2
    first, second = prompt[:midpoint], prompt[midpoint:]
    return second + first


@register_prompt_variant("reverse_sentences")
def variant_reverse_sentences(prompt: str, _: Dict[str, object]) -> str:
    sentences = split_into_sentences(prompt)
    sentences.reverse()
    return " ".join(sentences)


@register_prompt_variant("shuffle_sentences")
def variant_shuffle_sentences(prompt: str, options: Dict[str, object]) -> str:
    sentences = split_into_sentences(prompt)
    rng = random.Random(options.get("seed", 0))
    rng.shuffle(sentences)
    return " ".join(sentences)


@register_prompt_variant("symmetric_concat")
def variant_symmetric_concat(prompt: str, options: Dict[str, object]) -> str:
    separator = options.get("separator", "\n---\n")
    return f"{prompt}{separator}{prompt[::-1]}"


@register_prompt_variant("base_prefixed")
def variant_base_prefixed(prompt: str, options: Dict[str, object]) -> str:
    prefix = options.get("prefix", "[BASE] ")
    return f"{prefix}{prompt}"


@register_prompt_variant("sft_prefixed")
def variant_sft_prefixed(prompt: str, options: Dict[str, object]) -> str:
    prefix = options.get("prefix", "[SFT] ")
    return f"{prefix}{prompt}"


@register_prompt_variant("dual_channel")
def variant_dual_channel(prompt: str, options: Dict[str, object]) -> str:
    """
    Creates an asymmetrical prompt by providing separate base/SFT prefices.
    """
    base_prefix = options.get("base_prefix", "BASE INPUT:\n")
    sft_prefix = options.get("sft_prefix", "SFT INPUT:\n")
    order = options.get("order", "base_first")
    base_section = f"{base_prefix}{prompt}"
    sft_section = f"{sft_prefix}{prompt}"
    if order == "sft_first":
        return f"{sft_section}\n\n{base_section}"
    return f"{base_section}\n\n{sft_section}"


def generate_variants(
    prompt: str,
    variant_names: Iterable[str],
    *,
    variant_options: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, str]:
    """
    Produce multiple prompt variants at once.
    """
    outputs: Dict[str, str] = {}
    for name in variant_names:
        opts = variant_options.get(name, {}) if variant_options else {}
        outputs[name] = REGISTRY.apply(name, prompt, opts)
    return outputs


__all__ = [
    "REGISTRY",
    "register_prompt_variant",
    "generate_variants",
    "split_into_sentences",
]


