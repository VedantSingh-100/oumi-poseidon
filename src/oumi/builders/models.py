from pathlib import Path
from typing import Optional, Union, cast

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from oumi.core.configs import ModelParams, PeftParams
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
    find_model_hf_config,
    get_all_vlms_map,
)
from oumi.core.distributed import get_device_rank_info
from oumi.core.registry import REGISTRY, RegistryType
from oumi.core.tokenizers import get_default_special_tokens
from oumi.utils.distributed_utils import is_using_accelerate_fsdp
from oumi.utils.io_utils import get_oumi_root_directory, load_file
from oumi.utils.logging import logger
from oumi.utils.torch_naming_heuristics import disable_dropout
from oumi.utils.torch_utils import freeze_model_layers

try:
    import liger_kernel.transformers  # type: ignore
except ImportError:
    liger_kernel = None


def build_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds and returns a model based on the provided Oumi configuration.

    Args:
        model_params: The model parameters.
        peft_params: The PEFT parameters.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    if REGISTRY.contains(name=model_params.model_name, type=RegistryType.MODEL):
        model = build_oumi_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )
    elif model_params.model_name in (
        "nyu-visionx/cambrian-phi3-3b",
        "nyu-visionx/cambrian-8b",
        "nyu-visionx/cambrian-13b",
        "nyu-visionx/cambrian-34b",
    ):
        model = build_cambrian_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )
    else:
        model = build_huggingface_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )

    if model_params.enable_liger_kernel:
        _patch_model_for_liger_kernel(model)

    if len(model_params.freeze_layers) > 0:
        num_frozen = freeze_model_layers(model, model_params.freeze_layers)
        logger.warning(
            f"{num_frozen} layer(s) frozen based on the config: "
            f"{model_params.freeze_layers}."
        )

    if model_params.compile:
        # The output type of torch.compile is Callable, but when I test it it's of type
        # nn.Module. We cast it so that this function can have a useful return type.
        model = cast(nn.Module, torch.compile(model))
        logger.info("Enabled model compilation.")

    return model


def _get_model_type(model: nn.Module) -> Optional[str]:
    return getattr(model, "config", None) and getattr(model.config, "model_type", None)


def _patch_model_for_liger_kernel(model: nn.Module) -> None:
    """Patches the model for Liger Kernel.

    The list of support models can be found here:
    https://github.com/linkedin/Liger-Kernel/blob/99599091373f178e8ad6a69ecb1b32351d1d5c1f/src/liger_kernel/transformers/monkey_patch.py#L700

    If the model is not supported, liger kernel patching will not be applied,
    and a warning will be logged.
    """
    if liger_kernel is None:
        raise ImportError(
            "Liger Kernel not installed. Please install `pip install liger-kernel`."
        )

    model_type = _get_model_type(model)

    if model_type is None:
        raise ValueError(f"Could not find model type for: {model}")

    liger_kernel.transformers._apply_liger_kernel(model_type)


def build_oumi_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds a custom model from our Oumi registry."""
    model_class = REGISTRY[model_params.model_name, RegistryType.MODEL]
    model = model_class(model_params=model_params, **model_params.model_kwargs)

    # if model_params.load_pretrained_weights:
    #     raise NotImplementedError

    # if peft_params and peft_params.q_lora:
    #     raise NotImplementedError

    # if model_params.adapter_model is not None:
    #     raise NotImplementedError

    dtype = model_params.torch_dtype
    model = model.to(dtype=dtype)
    # Needed for MFUTrainerCallback
    # model.dtype = dtype
    return model


def build_huggingface_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds a HuggingFace model.

    If a local directory is specified, the model will be loaded from that checkpoint.
    Otherwise, `model_params.model_name` is the name of a HuggingFaceHub model. The
    model will be downloaded from the Hub to a local cache directory if it is not
    already present, and will be loaded from there.
    """
    device_map = model_params.device_map
    device_rank_info = get_device_rank_info()

    # If we're using FSDP via HF Accelerate, we should not specify the device map
    # so that HF properly initializes the model for FSDP.
    # If we set device_map to "auto", it seems HF will try to shard the model when
    # loading it, which conflicts with FSDP's sharding.
    # If we set device_map to f"cuda:{device_rank_info.local_rank}", it will try to
    # load the model only on rank 0, which will OOM for large models.
    # See https://github.com/huggingface/transformers/pull/25107.
    if is_using_accelerate_fsdp():
        logger.info("Accelerate FSDP run detected! Setting device_map to None.")
        device_map = None
    elif device_map == "auto" and device_rank_info.world_size > 1:
        # "auto" is not compatible with DDP.
        logger.info(
            f"Building model for distributed training "
            f"(world_size: {device_rank_info.world_size})..."
        )
        device_map = f"cuda:{device_rank_info.local_rank}"
    logger.info(
        f"Building model using device_map: {device_map} ({device_rank_info})..."
    )

    hf_config = find_model_hf_config(
        model_params.model_name, trust_remote_code=model_params.trust_remote_code
    )

    # (Experimental) Detects dropout probabilities in config and sets them to 0.0.
    if model_params.model_kwargs.get("disable_dropout"):
        disable_dropout(hf_config)
        del model_params.model_kwargs["disable_dropout"]

    if peft_params and peft_params.q_lora:
        quantization_config = peft_params.to_bits_and_bytes()
    else:
        quantization_config = None

    # Both functions instantiate a model from the config, but the main difference is
    # `load_pretrained_weights` also loads the weights, and `from_config` initializes
    # the weights from scratch based on the params in the config and the model class.
    transformers_model_class = _get_transformers_model_class(hf_config)

    if model_params.load_pretrained_weights:
        model = transformers_model_class.from_pretrained(
            config=hf_config,
            torch_dtype=model_params.torch_dtype,
            device_map=device_map,
            trust_remote_code=model_params.trust_remote_code,
            pretrained_model_name_or_path=model_params.model_name,
            quantization_config=quantization_config,
            attn_implementation=model_params.attn_implementation,
            **kwargs,
        )
    else:
        model = transformers_model_class.from_config(
            config=hf_config,
            torch_dtype=model_params.torch_dtype,
            trust_remote_code=model_params.trust_remote_code,
            attn_implementation=model_params.attn_implementation,
            **kwargs,
        )

    # Required for FSDP.
    # Context: https://github.com/huggingface/transformers/issues/28499
    model.config.use_cache = False

    # TODO Find a better way to handle it

    # Load pretrained PEFT adapters
    if model_params.adapter_model:
        logger.info(f"Loading PEFT adapter from: {model_params.adapter_model} ...")
        model = PeftModel.from_pretrained(model, model_params.adapter_model)

    return model


def _get_transformers_model_class(config):
    vlm_info = get_all_vlms_map().get(config.model_type, None)

    if vlm_info is not None:
        auto_model_class = vlm_info.model_class
        if not vlm_info.tested:
            logger.warning(
                f"Model type {config.model_type} not tested. "
                f"Using {auto_model_class} as the model class. "
                "If you encounter errors, please open an issue at https://github.com/oumi-ai/oumi."
            )
    else:
        auto_model_class = transformers.AutoModelForCausalLM
    logger.info(f"Using model class: {auto_model_class} to instantiate model.")
    return auto_model_class


def is_image_text_llm_using_model_name(
    model_name: str, trust_remote_code: bool
) -> bool:
    """Determines whether the model is a basic image+text LLM."""
    model_config = find_internal_model_config_using_model_name(
        model_name, trust_remote_code=trust_remote_code
    )
    return model_config is not None and model_config.visual_config is not None


def is_image_text_llm(model_params: ModelParams) -> bool:
    """Determines whether the model is a basic image+text LLM."""
    # For now, assume that custom models are not image+text LLMs.
    if REGISTRY.contains(name=model_params.model_name, type=RegistryType.MODEL):
        return False
    return is_image_text_llm_using_model_name(
        model_params.model_name, model_params.trust_remote_code
    )


def build_cambrian_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Downloads and builds the model from the HuggingFace Hub."""
    from importlib.util import find_spec

    for dependency_name in ("diffusers", "einops", "open_clip", "timm"):
        if not find_spec(dependency_name):
            raise RuntimeError(
                f"Failed to find the required dependency package:'{dependency_name}' "
                f"for the Cambrian model: '{model_params.model_name}'. "
                "Run `pip install oumi[cambrian]`, and try again."
            )

    try:
        from oumi.models.experimental.cambrian.mm_utils import (
            get_model_name_from_path as get_cambrian_model_name_from_path,
        )
        from oumi.models.experimental.cambrian.model.builder import (
            load_pretrained_model as load_cambrian_pretrained_model,
        )
    except ImportError as e:
        raise RuntimeError(
            "Failed to load a required dependency "
            f"for the Cambrian model: '{model_params.model_name}'. "
            "Run `pip install oumi[cambrian]`, and try again."
        ) from e

    device_map = model_params.device_map
    device_rank_info = get_device_rank_info()

    # If we're using FSDP via HF Accelerate, we should not specify the device map
    # so that HF properly initializes the model for FSDP.
    # If we set device_map to "auto", it seems HF will try to shard the model when
    # loading it, which conflicts with FSDP's sharding.
    # If we set device_map to f"cuda:{device_rank_info.local_rank}", it will try to
    # load the model only on rank 0, which will OOM for large models.
    # See https://github.com/huggingface/transformers/pull/25107.
    if is_using_accelerate_fsdp():
        logger.info("Accelerate FSDP run detected! Setting device_map to None.")
        device_map = None
    elif device_map == "auto" and device_rank_info.world_size > 1:
        # "auto" is not compatible with DDP.
        logger.info(
            f"Building model for distributed training "
            f"(world_size: {device_rank_info.world_size})..."
        )
        device_map = f"cuda:{device_rank_info.local_rank}"
    logger.info(
        f"Building model using device_map: {device_map} ({device_rank_info})..."
    )

    model_path = str(Path(model_params.model_name).expanduser())
    model_name = get_cambrian_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_cambrian_pretrained_model(
        model_path, None, model_name, device_map=(device_map or "auto")
    )

    # Required for FSDP.
    # Context: https://github.com/huggingface/transformers/issues/28499
    model.config.use_cache = False

    # TODO Find a better way to handle it

    # Load pretrained PEFT adapters
    if model_params.adapter_model:
        logger.info(f"Loading PEFT adapter from: {model_params.adapter_model} ...")
        model = PeftModel.from_pretrained(model, model_params.adapter_model)

    return model

def build_tokenizer(
    model_params: ModelParams,
) -> Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast, transformers.AutoProcessor]:
    """Builds and returns a tokenizer based on the provided Oumi configuration.

    Args:
        model_params (ModelParams): The model parameters.

    Returns:
        tokenizer: The tokenizer object built from the configuration.
    """
    # Identify the tokenizer we need to leverage for this model.
    if model_params.tokenizer_name:
        tokenizer_name = model_params.tokenizer_name
    else:
        # If no specific tokenizer is defined, fall back to model's default.
        tokenizer_name = model_params.model_name

    # Download and build the tokenizer from the HuggingFace Hub.
    if "poseidon_pde" in model_params.model_name:
        print("Inside the camlab-ethz function mentioned")
        return transformers.AutoImageProcessor.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            trust_remote_code=model_params.trust_remote_code,
            **model_params.tokenizer_kwargs,
        )
    else:
        print("Inside the autotokenizer function")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=model_params.trust_remote_code,
            **model_params.tokenizer_kwargs,
        )

    if model_params.tokenizer_pad_token:
        tokenizer.add_special_tokens(
            special_tokens_dict={"pad_token": model_params.tokenizer_pad_token}
        )

    # Ensure that the tokenizer has a pad token set.
    if (tokenizer.pad_token is None) and (tokenizer.pad_token_id is None):
        default_pad_token = get_default_special_tokens(tokenizer).pad_token
        if default_pad_token:
            logger.warning(f"Undefined pad token. Setting it to `{default_pad_token}`.")
            tokenizer.add_special_tokens(
                special_tokens_dict={"pad_token": default_pad_token}
            )
        else:
            raise ValueError(
                "Tokenizer does not have a pad token. This is expected for older "
                "models, but you need to set it manually in your model config as: "
                "tokenizer_kwargs={'pad_token': 'user_defined_pad_token'}"
            )

    if model_params.model_max_length:
        tokenizer.model_max_length = model_params.model_max_length

    if model_params.chat_template:
        logger.info(
            f"Using the chat template '{model_params.chat_template}' "
            "specified in model config!"
        )
        tokenizer.chat_template = build_chat_template(model_params.chat_template)

    if tokenizer.chat_template is None:
        logger.warning(
            "No chat template found for tokenizer. "
            "Please specify a chat template using the `chat_template` field. "
            "This will be required in future versions of Oumi."
        )
        logger.warning(
            "Setting tokenizer to use the 'default' chat template. "
            "The 'default' template does not use any special tokens, "
            "and is unlikely to yield good results. "
        )
        tokenizer.chat_template = build_chat_template(template_name="default")

    return tokenizer


def build_peft_model(
    base_model, use_gradient_checkpointing: bool, peft_params: PeftParams
):
    """Builds a PEFT model based on the given base model and params.

    Args:
        base_model: The base model to build the PEFT model on.
        use_gradient_checkpointing: Enable/disable gradient checkpointing.
        peft_params: The desired params for LORA.

    Returns:
        The built PEFT model.
    """
    lora_config = LoraConfig(
        r=peft_params.lora_r,
        lora_alpha=peft_params.lora_alpha,
        lora_dropout=peft_params.lora_dropout,
        target_modules=peft_params.lora_target_modules,
        modules_to_save=peft_params.lora_modules_to_save,
        bias=peft_params.lora_bias,  # type: ignore
        task_type=peft_params.lora_task_type,
    )

    if peft_params.q_lora:
        model = prepare_model_for_kbit_training(
            model=base_model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        model = base_model

    model = get_peft_model(model, lora_config)

    return model


def build_chat_template(template_name: str) -> str:
    """Builds a chat template based on code name.

    Args:
        template_name: the code name describing the chat-template.

    Raises:
        FileNotFoundError: if the requested template file does not exist.

    Returns:
        str: a jinja-based chat-template.
    """
    chat_template_directory = get_oumi_root_directory() / "datasets" / "chat_templates"

    template_file = f"{template_name.lower()}.jinja"
    chat_template_file = chat_template_directory / template_file

    if not chat_template_file.exists():
        existing_templates = [f.stem for f in chat_template_directory.glob("*.jinja")]
        error_message = (
            f"Chat template file not found: {chat_template_file}\n"
            f"Existing templates: {', '.join(existing_templates)}\n"
            f"To add a new template, create a .jinja file in {chat_template_directory}"
        )
        raise FileNotFoundError(error_message)

    chat_template = load_file(chat_template_file)

    # Remove indentation and newlines while preserving intentional whitespace
    lines = chat_template.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    chat_template = "".join(cleaned_lines)

    return chat_template
