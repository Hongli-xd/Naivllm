import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig
import re


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


def _load_into_param(param, tensors, ids):
    """Load ``tensors`` into ``param``, honoring its TP-aware ``weight_loader``.

    ``ids`` carries the per-tensor identifier expected by merged projections
    ('q'/'k'/'v' for QKV, 0/1 for gate_up); None for plain parameters.
    """
    loader = getattr(param, "weight_loader", None)
    if callable(loader):
        for tensor, tensor_id in zip(tensors, ids):
            if tensor_id is None:
                loader(param, tensor)
            else:
                loader(param, tensor, tensor_id)
        return
    if len(tensors) == 1:
        if param.shape != tensors[0].shape:
            # vocab-size mismatch fallback (trimmed checkpoints)
            min_size = min(param.shape[0], tensors[0].shape[0])
            param.data[:min_size].copy_(tensors[0][:min_size])
        else:
            param.data.copy_(tensors[0])
    else:
        param.data.copy_(torch.cat(tensors, dim=0))


def k_or_v_present(hf_weights: dict, q_name: str, suffix: str) -> bool:
    """Whether the k/v counterparts of a q_proj tensor exist in the checkpoint."""
    k_name = q_name.replace('q_proj', 'k_proj')
    v_name = q_name.replace('q_proj', 'v_proj')
    return k_name in hf_weights and v_name in hf_weights


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download

    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None

    # First, try local paths
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        # 用safe_open的内存映射，避免一次性加载大文件
        # 现代大模型通常将权重分片存储，聚合后提供统一的字典访问
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Now map and load weights into custom model
    loaded_params = set()
    skipped_params = []

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        try:
            layer_match = re.search(r'layers\.(\d+)', hf_name)
            layer_idx = layer_match.group(1) if layer_match else None

            # 1. Handle QKV merge (q_proj + k_proj + v_proj → qkv_projection),
            #    for both weight and bias (Qwen3 checkpoints with
            #    attention_bias=True carry q/k/v biases)
            if (
                layer_idx is not None
                and '.self_attn.q_proj.weight' in hf_name
                and k_or_v_present(hf_weights, hf_name, '.weight')
            ):
                k_name = hf_name.replace('q_proj', 'k_proj')
                v_name = hf_name.replace('q_proj', 'v_proj')
                custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                try:
                    param = model.get_parameter(custom_name)
                    _load_into_param(param,
                                     [hf_weight, hf_weights[k_name], hf_weights[v_name]],
                                     ['q', 'k', 'v'])
                    loaded_params.update({custom_name, hf_name, k_name, v_name})
                except AttributeError:
                    skipped_params.append((custom_name, "Parameter not found"))

            elif (
                layer_idx is not None
                and '.self_attn.q_proj.bias' in hf_name
                and k_or_v_present(hf_weights, hf_name, '.bias')
            ):
                k_name = hf_name.replace('q_proj', 'k_proj')
                v_name = hf_name.replace('q_proj', 'v_proj')
                custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.bias"
                try:
                    param = model.get_parameter(custom_name)
                    _load_into_param(param,
                                     [hf_weight, hf_weights[k_name], hf_weights[v_name]],
                                     ['q', 'k', 'v'])
                    loaded_params.update({custom_name, hf_name, k_name, v_name})
                except AttributeError:
                    skipped_params.append((custom_name, "Parameter not found"))

            # 2. Handle gate_up merge (gate_proj + up_proj → gate_up)
            elif (
                layer_idx is not None
                and '.mlp.gate_proj.weight' in hf_name
                and hf_name.replace('gate_proj', 'up_proj') in hf_weights
            ):
                up_name = hf_name.replace('gate_proj', 'up_proj')
                custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                try:
                    param = model.get_parameter(custom_name)
                    _load_into_param(param, [hf_weight, hf_weights[up_name]], [0, 1])
                    loaded_params.update({custom_name, hf_name, up_name})
                except AttributeError:
                    skipped_params.append((custom_name, "Parameter not found"))

            # 3. Handle gate_up merge for bias (if present)
            elif (
                layer_idx is not None
                and '.mlp.gate_proj.bias' in hf_name
                and hf_name.replace('gate_proj', 'up_proj') in hf_weights
            ):
                up_bias_name = hf_name.replace('gate_proj', 'up_proj')
                custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                try:
                    param = model.get_parameter(custom_name)
                    _load_into_param(param, [hf_weight, hf_weights[up_bias_name]], [0, 1])
                    loaded_params.update({custom_name, hf_name, up_bias_name})
                except AttributeError:
                    skipped_params.append((custom_name, "Parameter not found"))

            # 4. Skip k_proj, v_proj, up_proj (already merged)
            elif any(x in hf_name for x in ['.k_proj.', '.v_proj.', '.up_proj.']):
                if hf_name not in loaded_params:
                    skipped_params.append((hf_name, "Merged into qkv_projection or gate_up"))

            # 5. All other parameters: load through the parameter's own
            #    (possibly TP-sharding) weight_loader
            else:
                try:
                    param = model.get_parameter(hf_name)
                    _load_into_param(param, [hf_weight], [None])
                    loaded_params.add(hf_name)
                except AttributeError:
                    skipped_params.append((hf_name, "Parameter not found"))

        except Exception as e:
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # Check for model parameters that weren't loaded
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")

    if unloaded_params:
        print(f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f"  ... and {len(unloaded_params) - 15} more")

    if skipped_params:
        # Group skipped by reason
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]

        if merged_skips:
            print(f"Skipped (merged into other weights): {len(merged_skips)}")
        if not_found_skips:
            print(f"Skipped (not found in model): {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:
                print(f"  - {name}")
            if len(not_found_skips) > 5:
                print(f"  ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:
            print(f"Skipped (no mapping rule): {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:
                print(f"  - {name}")
            if len(no_mapping_skips) > 5:
                print(f"  ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")
    return loaded_params
