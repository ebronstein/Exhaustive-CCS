"""Utilities for working with Hugging Face models and tokenizers.

From EleutherAI/ccs repository.
"""

import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
)

# Ordered by preference
_DECODER_ONLY_SUFFIXES = [
    "CausalLM",
    "LMHeadModel",
]
# Includes encoder-decoder models
_AUTOREGRESSIVE_SUFFIXES = ["ConditionalGeneration"] + _DECODER_ONLY_SUFFIXES


@contextmanager
def prevent_name_conflicts():
    """Temporarily change cwd to a temporary directory, to prevent name conflicts."""
    with TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            yield
        finally:
            os.chdir(old_cwd)


def get_model_class(model_name: str) -> str:
    if "gpt-neo-2.7B" in model_name or "gpt-j-6B" in model_name:
        model = AutoModelForCausalLM
    elif "gpt2" in model_name:
        model = GPT2LMHeadModel
    elif "T0" in model_name:
        model = AutoModelForSeq2SeqLM
    elif "unifiedqa" in model_name:
        model = T5ForConditionalGeneration
    elif "deberta" in model_name:
        model = AutoModelForSequenceClassification
    elif "roberta" in model_name:
        model = AutoModelForSequenceClassification
    elif "t5" in model_name:
        model = AutoModelWithLMHead
    else:
        model = AutoModel

    return model


def get_tokenizer_class(model_name: str) -> str:
    if "gpt2" in model_name:
        tokenizer = GPT2Tokenizer
    else:
        tokenizer = AutoTokenizer

    return tokenizer


def instantiate_model(
    model_str: str,
    device: Union[str, torch.device] = "cpu",
    use_auth_token: Optional[Union[bool, str]] = None,
    **kwargs,
) -> PreTrainedModel:
    """Instantiate a model string with the appropriate `Auto` class.

    Args:
        model_str (str): Full model name. This should match the name in the
            Hugging Face model hub.
        device (Union[str, torch.device], optional): The device to use for the model. Defaults to "cpu".
        use_auth_token (Optional[Union[bool, str]], optional): Optional authentication token for accessing models on the Hugging Face Hub. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the model instantiation.

    Returns:
        PreTrainedModel: The instantiated model.

    Raises:
        ValueError: If attempting to load in 8-bit weights with fp32 weights.
    """
    device = torch.device(device)
    kwargs["device_map"] = "auto"

    with prevent_name_conflicts():
        model_cfg = AutoConfig.from_pretrained(model_str, use_auth_token=use_auth_token)

        # When the torch_dtype is None, this generally means the model is fp32, because
        # the config was probably created before the `torch_dtype` field was added.
        fp32_weights = model_cfg.torch_dtype in (None, torch.float32)

        # Required by `bitsandbytes` to load in 8-bit.
        if kwargs.get("load_in_8bit"):
            # Sanity check: we probably shouldn't be loading in 8-bit if the checkpoint
            # is in fp32. `bitsandbytes` only supports mixed fp16/int8 inference, and
            # we can't guarantee that there won't be overflow if we downcast to fp16.
            if fp32_weights:
                raise ValueError("Cannot load in 8-bit if weights are fp32")

            kwargs["torch_dtype"] = torch.float16

        # CPUs generally don't support anything other than fp32.
        elif device.type == "cpu":
            kwargs["torch_dtype"] = torch.float32

        # If the model is fp32 but bf16 is available, convert to bf16.
        # Usually models with fp32 weights were actually trained in bf16, and
        # converting them doesn't hurt performance.
        elif fp32_weights and torch.cuda.is_bf16_supported():
            kwargs["torch_dtype"] = torch.bfloat16
            print("Weights seem to be fp32, but bf16 is available. Loading in bf16.")
        else:
            kwargs["torch_dtype"] = "auto"

        archs = model_cfg.architectures
        if not isinstance(archs, list):
            return AutoModel.from_pretrained(model_str, **kwargs)

        for suffix in _AUTOREGRESSIVE_SUFFIXES:
            # Check if any of the architectures in the config end with the suffix.
            # If so, return the corresponding model class.
            for arch_str in archs:
                if arch_str.endswith(suffix):
                    model_cls = getattr(transformers, arch_str)
                    return model_cls.from_pretrained(model_str, **kwargs)

        model_cls = get_model_class(model_str)
        return model_cls.from_pretrained(model_str, **kwargs)


def instantiate_tokenizer(model_str: str, **kwargs) -> PreTrainedTokenizerBase:
    """Instantiate a tokenizer, using the fast one iff it exists."""
    with prevent_name_conflicts():
        tokenizer_cls = get_tokenizer_class(model_str)
        try:
            return tokenizer_cls.from_pretrained(model_str, use_fast=True, **kwargs)
        except Exception as e:
            if kwargs.get("verbose", True):
                print(f"Falling back to slow tokenizer; fast one failed: '{e}'")

            return tokenizer_cls.from_pretrained(model_str, use_fast=False, **kwargs)


def get_model_config(
    model_str: str, use_auth_token: Optional[Union[bool, str]] = None
) -> PretrainedConfig:
    """Return the model config for a given model."""
    return AutoConfig.from_pretrained(model_str, use_auth_token=use_auth_token)


def is_autoregressive(model_cfg: PretrainedConfig, include_enc_dec: bool) -> bool:
    """Check if a model config is autoregressive."""
    archs = model_cfg.architectures
    if not isinstance(archs, list):
        return False

    suffixes = _AUTOREGRESSIVE_SUFFIXES if include_enc_dec else _DECODER_ONLY_SUFFIXES
    return any(arch_str.endswith(suffix) for arch_str in archs for suffix in suffixes)


def is_decoder_only(
    model_str: str,
    model_cfg: Optional[PretrainedConfig] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> bool:
    """Check if a model config is decoder-only."""
    if model_cfg is None:
        model_cfg = get_model_config(model_str, use_auth_token=use_auth_token)
    # For some reason, t5-11b is appears as a decoder-only architecture even
    # though it's encoder-decoder.
    return is_autoregressive(model_cfg, include_enc_dec=False) and model_str not in [
        "google-t5/t5-11b",
        "t5-11b",
    ]


def is_encoder_only(
    model_str: str,
    model_cfg: Optional[PretrainedConfig] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> bool:
    """Check if a model config is encoder-only."""
    if model_cfg is None:
        model_cfg = get_model_config(model_str, use_auth_token=use_auth_token)
    return not is_autoregressive(model_cfg, include_enc_dec=True)


def get_num_hidden_layers(
    model: str,
    use_auth_token: Optional[Union[bool, str]] = None,
) -> int:
    """Return the number of hidden layers in a model."""
    # Look up the model config to get the number of layers
    return AutoConfig.from_pretrained(
        model, use_auth_token=use_auth_token
    ).num_hidden_layers
