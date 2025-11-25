"""
Utilities to compare residual streams between a base model and its SFT variant.

This module focuses on Hugging Face causal decoder models. It projects residual
states through the (optionally swapped) unembedding matrix to expose logit-lens
statistics and quantifies divergence between models layer-by-layer.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


ModelKey = str  # Either "base" or "sft"
SOURCE_CHOICES = ("base", "sft")


def _resolve_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return None
    map_ = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if isinstance(dtype, str):
        lowered = dtype.lower()
        if lowered not in map_:
            raise ValueError(f"Unsupported dtype string: {dtype}. Expected one of {list(map_.keys()) + ['auto']}")
        return map_[lowered]
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unsupported dtype value: {dtype}")


@dataclass
class ModelSwapSpec:
    """Defines which embedding/unembedding weights to use for a forward pass."""

    embedding_source: ModelKey = "self"
    unembedding_source: ModelKey = "self"

    def normalize(self, default_source: ModelKey) -> "ModelSwapSpec":
        emb_src = self.embedding_source if self.embedding_source != "self" else default_source
        unemb_src = self.unembedding_source if self.unembedding_source != "self" else default_source
        if emb_src not in SOURCE_CHOICES:
            raise ValueError(f"embedding_source must be one of {SOURCE_CHOICES + ('self',)}")
        if unemb_src not in SOURCE_CHOICES:
            raise ValueError(f"unembedding_source must be one of {SOURCE_CHOICES + ('self',)}")
        return ModelSwapSpec(embedding_source=emb_src, unembedding_source=unemb_src)


def _device_from_str(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def _temporary_weight_swap(
    parameter: torch.nn.Parameter,
    swap_tensor: torch.Tensor,
    restore_tensor: torch.Tensor,
) -> Iterable[None]:
    """Temporarily copies swap_tensor into parameter and restores afterwards."""
    if swap_tensor.device != parameter.device or swap_tensor.dtype != parameter.dtype:
        swap_tensor = swap_tensor.to(device=parameter.device, dtype=parameter.dtype)
    if restore_tensor.device != parameter.device or restore_tensor.dtype != parameter.dtype:
        restore_tensor = restore_tensor.to(device=parameter.device, dtype=parameter.dtype)

    with torch.no_grad():
        parameter.data.copy_(swap_tensor)
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.data.copy_(restore_tensor)


class ResidualComparisonRunner:
    """
    Runs paired forwards on base and SFT Hugging Face causal decoder models and
    summarizes per-layer residual/logit statistics.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        sft_model_name_or_path: str,
        *,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        top_k: int = 20,
        tracked_token_ids: Optional[List[int]] = None,
        tracked_token_strings: Optional[List[str]] = None,
        interesting_token_map: Optional[Dict[str, str]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = _device_from_str(device)
        self.dtype = _resolve_dtype(dtype)
        self.top_k = top_k
        self.model_kwargs = model_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or sft_model_name_or_path,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            # Fallback required for batch collation
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = self._load_model(base_model_name_or_path)
        self.sft_model = self._load_model(sft_model_name_or_path)

        self.base_model.eval()
        self.sft_model.eval()

        self.tie_word_embeddings = self.base_model.config.tie_word_embeddings
        if self.sft_model.config.tie_word_embeddings != self.tie_word_embeddings:
            raise ValueError("Base and SFT models disagree on tie_word_embeddings flag.")

        # Cache embedding/unembedding weights for swapping and logit lens projections.
        self.param_snapshots: Dict[ModelKey, Dict[str, torch.Tensor]] = {
            "base": self._snapshot_params(self.base_model),
            "sft": self._snapshot_params(self.sft_model),
        }

        self.embedding_layers: Dict[ModelKey, torch.nn.Module] = {
            "base": self.base_model.get_input_embeddings(),
            "sft": self.sft_model.get_input_embeddings(),
        }

        self.default_tracked_ids = set(tracked_token_ids or [])
        if tracked_token_strings:
            ids_from_strings = self.tokenizer.convert_tokens_to_ids(tracked_token_strings)
            self.default_tracked_ids.update(id_ for id_ in ids_from_strings if id_ != self.tokenizer.unk_token_id)

        self.interesting_token_ids = self._resolve_interesting_tokens(interesting_token_map or {})

    def _load_model(self, name_or_path: str) -> PreTrainedModel:
        kwargs = dict(
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        kwargs.update(self.model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        model.to(self.device)
        return model

    def _snapshot_params(self, model: PreTrainedModel) -> Dict[str, torch.Tensor]:
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()
        if input_emb is None or output_emb is None:
            raise ValueError("Model must expose both input and output embeddings for comparison.")

        return {
            "embedding": input_emb.weight.detach().cpu().clone(),
            "unembedding": output_emb.weight.detach().cpu().clone(),
        }

    def _resolve_interesting_tokens(self, token_map: Dict[str, str]) -> Dict[str, int]:
        """
        Convert a dict like {"eos": ""} or {"refusal": "I'm sorry"} into token ids.
        Values are token strings (single token preferred).
        """
        resolved: Dict[str, int] = {}
        for name, token in token_map.items():
            token_id = None
            if isinstance(token, int):
                token_id = token
            else:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                continue
            resolved[name] = token_id
        return resolved

    def compare_prompt(
        self,
        prompt: str,
        *,
        swap_options: Optional[Dict[ModelKey, ModelSwapSpec]] = None,
        tracked_token_ids: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare base vs SFT models on a single prompt and return JSON-serializable stats.
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device) if "attention_mask" in encoded else None
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        swap_options = swap_options or {}
        base_spec = swap_options.get("base", ModelSwapSpec()).normalize("base")
        sft_spec = swap_options.get("sft", ModelSwapSpec()).normalize("sft")

        tracked_ids = set(self.default_tracked_ids)
        if tracked_token_ids:
            tracked_ids.update(tracked_token_ids)
        tracked_ids.update(self.interesting_token_ids.values())

        base_hidden = self._forward_with_hidden_states("base", self.base_model, input_ids, attention_mask, base_spec)
        sft_hidden = self._forward_with_hidden_states("sft", self.sft_model, input_ids, attention_mask, sft_spec)

        layer_stats = self._assemble_stats(
            base_hidden=base_hidden,
            sft_hidden=sft_hidden,
            tokens=tokens,
            tracked_ids=sorted(tracked_ids),
            spec_pair=(base_spec, sft_spec),
        )

        return {
            "prompt": prompt,
            "tokens": tokens,
            "metadata": metadata or {},
            "swap_options": {
                "base": base_spec.__dict__,
                "sft": sft_spec.__dict__,
            },
            "layers": layer_stats,
        }

    def _forward_with_hidden_states(
        self,
        model_key: ModelKey,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        swap_spec: ModelSwapSpec,
    ) -> List[torch.Tensor]:
        """
        Runs a forward pass, optionally swapping the input embedding weights beforehand.
        Returns hidden states for each layer (including embedding layer at index 0).
        """
        embedding_layer = self.embedding_layers[model_key]
        swap_ctx = nullcontext()

        if swap_spec.embedding_source != model_key:
            source_snapshot = self.param_snapshots[swap_spec.embedding_source]["embedding"]
            restore_snapshot = self.param_snapshots[model_key]["embedding"]
            swap_ctx = _temporary_weight_swap(
                embedding_layer.weight,
                source_snapshot,
                restore_snapshot,
            )

        with torch.no_grad():
            with swap_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = list(outputs.hidden_states)  # type: ignore[attr-defined]
        return hidden_states

    def _assemble_stats(
        self,
        *,
        base_hidden: List[torch.Tensor],
        sft_hidden: List[torch.Tensor],
        tokens: List[str],
        tracked_ids: List[int],
        spec_pair: Tuple[ModelSwapSpec, ModelSwapSpec],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Build JSON-friendly stats keyed by layer -> position.
        """
        if len(base_hidden) != len(sft_hidden):
            raise ValueError("Base and SFT hidden state lengths differ; cannot compare.")
        num_layers = len(base_hidden)
        seq_len = base_hidden[0].shape[1]

        base_unembed = self.param_snapshots[spec_pair[0].unembedding_source]["unembedding"].to(self.device)
        sft_unembed = self.param_snapshots[spec_pair[1].unembedding_source]["unembedding"].to(self.device)

        layer_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for layer_idx in range(1, num_layers):  # skip embedding state at index 0
            base_layer = base_hidden[layer_idx][0]  # (seq, hidden)
            sft_layer = sft_hidden[layer_idx][0]
            prev_base_layer = base_hidden[layer_idx - 1][0]
            prev_sft_layer = sft_hidden[layer_idx - 1][0]

            layer_payload: Dict[str, Dict[str, Any]] = {}
            for pos in range(seq_len):
                layer_payload[str(pos)] = self._compute_position_stats(
                    base_res=base_layer[pos],
                    sft_res=sft_layer[pos],
                    prev_base_res=prev_base_layer[pos],
                    prev_sft_res=prev_sft_layer[pos],
                    tokens=tokens,
                    position=pos,
                    tracked_ids=tracked_ids,
                    base_unembed=base_unembed,
                    sft_unembed=sft_unembed,
                )
            layer_results[str(layer_idx - 1)] = layer_payload

        return layer_results

    def _compute_position_stats(
        self,
        *,
        base_res: torch.Tensor,
        sft_res: torch.Tensor,
        tokens: List[str],
        position: int,
        tracked_ids: List[int],
        base_unembed: torch.Tensor,
        sft_unembed: torch.Tensor,
        prev_base_res: Optional[torch.Tensor] = None,
        prev_sft_res: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistics for a single layer/position pair, including
        intra-model deltas versus the previous layer when available.
        """
        base_logits = torch.matmul(base_res, base_unembed.T)
        sft_logits = torch.matmul(sft_res, sft_unembed.T)
        logit_shift = sft_logits - base_logits

        top_k = min(self.top_k, base_logits.shape[-1])
        top_pos = torch.topk(logit_shift, k=top_k)
        top_neg = torch.topk(-logit_shift, k=top_k)

        base_log_probs = F.log_softmax(base_logits, dim=-1)
        sft_log_probs = F.log_softmax(sft_logits, dim=-1)

        base_probs = base_log_probs.exp()
        sft_probs = sft_log_probs.exp()

        entropy_base = -(base_probs * base_log_probs).sum().item()
        entropy_sft = -(sft_probs * sft_log_probs).sum().item()
        kl_div = (sft_probs * (sft_log_probs - base_log_probs)).sum().item()

        cosine_sim = F.cosine_similarity(base_res, sft_res, dim=0).item()
        norm_base = base_res.norm().item()
        norm_sft = sft_res.norm().item()
        norm_diff = (base_res - sft_res).norm().item()

        base_cosine_prev = None
        base_norm_delta = None
        if prev_base_res is not None:
            base_cosine_prev = F.cosine_similarity(base_res, prev_base_res, dim=0).item()
            base_norm_delta = norm_base - prev_base_res.norm().item()

        sft_cosine_prev = None
        sft_norm_delta = None
        if prev_sft_res is not None:
            sft_cosine_prev = F.cosine_similarity(sft_res, prev_sft_res, dim=0).item()
            sft_norm_delta = norm_sft - prev_sft_res.norm().item()

        tracked_logits = {}
        for token_id in tracked_ids:
            tracked_logits[str(token_id)] = {
                "base": base_logits[token_id].item(),
                "sft": sft_logits[token_id].item(),
            }

        return {
            "token": tokens[position] if position < len(tokens) else "",
            "top_k_increased": {
                "indices": top_pos.indices.tolist(),
                "values": [round(v, 6) for v in top_pos.values.tolist()],
            },
            "top_k_decreased": {
                "indices": top_neg.indices.tolist(),
                "values": [round(-v, 6) for v in top_neg.values.tolist()],
            },
            "entropy_base": entropy_base,
            "entropy_sft": entropy_sft,
            "kl_div": kl_div,
            "tracked_token_logits": tracked_logits,
            "cosine_sim": cosine_sim,
            "norm_base": norm_base,
            "norm_sft": norm_sft,
            "norm_diff": norm_diff,
            "base_cosine_prev": base_cosine_prev,
            "base_norm_delta": base_norm_delta,
            "sft_cosine_prev": sft_cosine_prev,
            "sft_norm_delta": sft_norm_delta,
        }


