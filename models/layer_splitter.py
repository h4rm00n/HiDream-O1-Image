import gc
import json
import os
import ctypes
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save


def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()


LAYER_GROUPS = {
    "vision": "model.visual.",
    "embed": [
        "model.language_model.embed_tokens.",
        "model.language_model.rotary_emb.",
    ],
    "decoder_layer_prefix": "model.language_model.layers.",
    "norm": "model.language_model.norm.",
    "head": [
        "model.t_embedder1.",
        "model.x_embedder.",
        "model.final_layer2.",
        "lm_head.",
    ],
}


def _load_weight_index(checkpoint_path: Path):
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        return index
    bin_index = checkpoint_path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index) as f:
            index = json.load(f)["weight_map"]
        return index
    single_sf = checkpoint_path / "model.safetensors"
    if single_sf.exists():
        state_dict = safetensors_load(str(single_sf), device="cpu")
        return {k: "model.safetensors" for k in state_dict}
    bin_files = sorted(checkpoint_path.glob("pytorch_model*.bin"))
    if bin_files:
        index = {}
        for bf in bin_files:
            sd = torch.load(str(bf), map_location="cpu")
            for k in sd:
                index[k] = bf.name
            del sd
        return index
    raise FileNotFoundError(
        f"No model weights found in {checkpoint_path}"
    )


def _count_decoder_layers(weight_map: dict) -> int:
    layer_indices = set()
    prefix = "model.language_model.layers."
    for k in weight_map:
        if k.startswith(prefix):
            idx_str = k[len(prefix):].split(".")[0]
            layer_indices.add(int(idx_str))
    return max(layer_indices) + 1 if layer_indices else 0


def check_space(checkpoint_path: Path, saving_path: Path):
    total_bytes = 0
    for f in checkpoint_path.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size
    free = shutil.disk_usage(saving_path).free
    if free < total_bytes * 1.1:
        free_gb = free / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(
            f"[WARNING] Only {free_gb:.1f}GB free, need ~{total_gb:.1f}GB. "
            f"Proceeding anyway..."
        )


def split_and_save_layers(
    checkpoint_path: str,
    layer_shards_saving_path: Optional[str] = None,
    delete_original: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    checkpoint_path = Path(checkpoint_path)
    if layer_shards_saving_path is None:
        saving_path = checkpoint_path / "splitted_layers"
    else:
        saving_path = Path(layer_shards_saving_path) / "splitted_layers"

    dtype_str = str(dtype) if dtype is not None else "original"
    done_file = saving_path / "_done"
    dtype_file = saving_path / "_dtype"

    if saving_path.exists() and done_file.exists():
        if dtype_file.exists():
            with open(dtype_file) as f:
                saved_dtype = f.read().strip()
            if saved_dtype == dtype_str:
                print(f"[split] Splitted layers already exist at {saving_path} (dtype={saved_dtype})")
                return str(saving_path)
            else:
                print(f"[split] dtype mismatch (saved={saved_dtype}, requested={dtype_str}), re-splitting...")
        else:
            print(f"[split] Splitted layers already exist at {saving_path}")
            return str(saving_path)

    saving_path.mkdir(parents=True, exist_ok=True)
    weight_map = _load_weight_index(checkpoint_path)
    n_layers = _count_decoder_layers(weight_map)

    safetensors_format = "model.safetensors.index.json" in [
        p.name for p in checkpoint_path.iterdir()
    ] or (checkpoint_path / "model.safetensors").exists()

    # Define layer groups
    groups = {}
    # Vision encoder
    groups["vision"] = [k for k in weight_map if k.startswith("model.visual.")]
    # Embed (embed_tokens + rotary_emb)
    groups["embed"] = [
        k for k in weight_map
        if k.startswith("model.language_model.embed_tokens.")
        or k.startswith("model.language_model.rotary_emb.")
    ]
    # Decoder layers
    for i in range(n_layers):
        prefix = f"model.language_model.layers.{i}."
        groups[f"decoder_layer_{i}"] = [k for k in weight_map if k.startswith(prefix)]
    # Norm
    groups["norm"] = [k for k in weight_map if k.startswith("model.language_model.norm.")]
    # Head (t_embedder, x_embedder, final_layer, lm_head)
    groups["head"] = [
        k for k in weight_map
        if k.startswith("model.t_embedder1.")
        or k.startswith("model.x_embedder.")
        or k.startswith("model.final_layer2.")
        or k.startswith("lm_head.")
    ]

    # Also include any remaining weights not captured above (buffers etc.)
    captured = set()
    for keys in groups.values():
        captured.update(keys)
    remaining = [k for k in weight_map if k not in captured]
    if remaining:
        groups["other"] = remaining

    check_space(checkpoint_path, saving_path)

    loaded_files = {}
    for group_name, group_keys in groups.items():
        if not group_keys:
            continue

        shard_files = set(weight_map[k] for k in group_keys)
        state_dict = {}

        for shard_file in shard_files:
            if shard_file not in loaded_files:
                file_path = checkpoint_path / shard_file
                if safetensors_format:
                    loaded_files[shard_file] = safetensors_load(str(file_path), device="cpu")
                else:
                    loaded_files[shard_file] = torch.load(str(file_path), map_location="cpu")

            for k in group_keys:
                if weight_map[k] == shard_file and k in loaded_files[shard_file]:
                    state_dict[k] = loaded_files[shard_file][k]

        sf_path = saving_path / f"{group_name}.safetensors"
        if dtype is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        safetensors_save(state_dict, sf_path)
        print(f"[split] Saved {len(group_keys)} params -> {sf_path}")

        # Free memory
        del state_dict
        clean_memory()

    # Write done marker
    (saving_path / "_done").touch()
    with open(saving_path / "_dtype", "w") as f:
        f.write(dtype_str)

    if delete_original:
        for shard_file in set(weight_map.values()):
            fp = checkpoint_path / shard_file
            if fp.exists():
                fp.unlink()

    print(f"[split] Done. Layers saved to {saving_path}")
    return str(saving_path)


def find_or_create_splitted_path(
    model_path: str,
    layer_shards_saving_path: Optional[str] = None,
    delete_original: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> tuple:
    """
    Returns (model_local_path, splitted_layers_path).
    If the model is already split, returns cached paths.
    """
    model_path = Path(model_path)

    if layer_shards_saving_path:
        splitted = Path(layer_shards_saving_path) / "splitted_layers"
    else:
        splitted = model_path / "splitted_layers"

    dtype_str = str(dtype) if dtype is not None else "original"
    done_file = splitted / "_done"
    dtype_file = splitted / "_dtype"

    if splitted.exists() and done_file.exists():
        if dtype_file.exists():
            with open(dtype_file) as f:
                saved_dtype = f.read().strip()
            if saved_dtype == dtype_str:
                return str(model_path), str(splitted)
            else:
                print(f"[split] dtype mismatch (saved={saved_dtype}, requested={dtype_str}), re-splitting...")
        else:
            # Old splits without dtype file - assume original precision
            if dtype is None:
                return str(model_path), str(splitted)
            else:
                print(f"[split] existing splits have no dtype info, re-splitting with {dtype_str}...")
                import shutil
                shutil.rmtree(splitted)

    splitted_path = split_and_save_layers(
        str(model_path),
        layer_shards_saving_path,
        delete_original,
        dtype,
    )
    return str(model_path), splitted_path
