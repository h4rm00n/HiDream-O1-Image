import gc
import ctypes
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from models.qwen3_vl_transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLModelOutputWithPast,
    Qwen3VLCausalLMOutputWithPast,
    apply_rotary_pos_emb,
)
from models.layer_splitter import find_or_create_splitted_path


_flash_attn_func = None
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
    except ImportError:
        pass


def _clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()


class LowVRAMImageGenerationModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        layer_shards_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype

        # Load config
        self.config = Qwen3VLConfig.from_pretrained(str(self.model_path), trust_remote_code=True)

        # Split model into per-layer files if needed
        mp, sp = find_or_create_splitted_path(str(self.model_path), layer_shards_path, dtype=self.dtype)
        self.splitted_path = Path(sp)
        self.model_local_path = mp

        # Create empty model skeleton
        self._init_empty_model()

        # Count decoder layers
        n_layers = self.config.text_config.num_hidden_layers
        self.num_decoder_layers = n_layers

        # Per-layer safetensors file listing
        self._decoder_layer_files = {
            i: self.splitted_path / f"decoder_layer_{i}.safetensors"
            for i in range(n_layers)
        }
        self._vision_file = self.splitted_path / "vision.safetensors"
        self._embed_file = self.splitted_path / "embed.safetensors"
        self._norm_file = self.splitted_path / "norm.safetensors"
        self._head_file = self.splitted_path / "head.safetensors"
        self._other_file = self.splitted_path / "other.safetensors"

        # Load and pin small modules permanently
        self._load_persistent_modules()

        # Track whether vision is currently loaded
        self._vision_loaded = False

    def _init_empty_model(self):
        with init_empty_weights():
            self.model = Qwen3VLForConditionalGeneration._from_config(self.config)
        self.model.eval()
        self.model.tie_weights()

        # Move buffers to device
        for n, buf in self.model.named_buffers():
            set_module_tensor_to_device(self.model, n, self.device, value=buf, dtype=self.dtype)

    def _load_safetensors(self, sf_path: Path) -> dict:
        return safetensors_load(str(sf_path), device="cpu")

    def _move_params_to_device(self, state_dict: dict):
        """Move a state_dict's params from CPU to GPU on the empty model."""
        for param_name, param in state_dict.items():
            try:
                param_dtype = self.dtype
                if param.dtype == torch.float8_e4m3fn:
                    param = param.to(torch.bfloat16)
                    param_dtype = torch.bfloat16
                set_module_tensor_to_device(
                    self.model, param_name, self.device, value=param, dtype=param_dtype,
                )
            except (ValueError, AttributeError, RuntimeError):
                pass

    def _move_params_to_meta(self, state_dict: dict):
        """Move params back to meta device to free GPU memory."""
        for param_name in state_dict:
            try:
                set_module_tensor_to_device(self.model, param_name, "meta")
            except (ValueError, AttributeError, RuntimeError):
                pass

    def _load_persistent_modules(self):
        """Load modules that stay on GPU throughout inference."""
        # Embed (embed_tokens + rotary_emb)
        if self._embed_file.exists():
            sd = self._load_safetensors(self._embed_file)
            self._move_params_to_device(sd)
            self._embed_state = sd
        else:
            self._embed_state = {}

        # Head (t_embedder1, x_embedder, final_layer2, lm_head)
        if self._head_file.exists():
            sd = self._load_safetensors(self._head_file)
            self._move_params_to_device(sd)
            self._head_state = sd
        else:
            self._head_state = {}

        # Norm
        if self._norm_file.exists():
            sd = self._load_safetensors(self._norm_file)
            self._move_params_to_device(sd)
            self._norm_state = sd
        else:
            self._norm_state = {}

        # Other (buffers, etc.)
        if self._other_file.exists():
            sd = self._load_safetensors(self._other_file)
            self._move_params_to_device(sd)
            self._other_state = sd
        else:
            self._other_state = {}

    def _load_vision_encoder(self):
        if self._vision_loaded:
            return
        if self._vision_file.exists():
            sd = self._load_safetensors(self._vision_file)
            self._move_params_to_device(sd)
            self._vision_state = sd
            self._vision_loaded = True

    def _unload_vision_encoder(self):
        if not self._vision_loaded:
            return
        self.model.visual.to("meta")
        self._vision_loaded = False
        _clean_memory()

    def _load_decoder_layer(self, layer_idx: int) -> dict:
        sf_path = self._decoder_layer_files[layer_idx]
        sd = self._load_safetensors(sf_path)
        self._move_params_to_device(sd)
        return sd

    def _unload_decoder_layer(self, state_dict: dict):
        self._move_params_to_meta(state_dict)
        _clean_memory()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value if isinstance(value, torch.device) else torch.device(value)

    # ------------------------------------------------------------------
    # Forward pass: image generation (vinputs != None)
    # ------------------------------------------------------------------

    def _forward_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
    ) -> Qwen3VLModelOutputWithPast:
        device = self.device
        dtype = self.dtype
        inner_model = self.model.model  # Qwen3VLModel

        # 1. Text token embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # 2. Image/video embeddings (vision encoder)
        if pixel_values is not None:
            self._load_vision_encoder()
            visual = inner_model.visual
            pixel_values = pixel_values.to(device=device, dtype=visual.dtype)
            image_embeds, _ = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(device, dtype)
            image_mask, _ = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            self._unload_vision_encoder()

        # 3. Embed timestep
        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(device)
        t_emb = inner_model.t_embedder1(timestep)  # [B, hidden]
        tms_mask = (input_ids == inner_model.tms_token_id)
        tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        t_emb_expanded = t_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

        # 4. Embed vinputs
        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(device)
        vinputs_embedded = inner_model.x_embedder(vinputs).to(dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, total_seq_len, _ = inputs_embeds.shape
        hidden_size = inputs_embeds.shape[-1]

        # 5. Parse token_types
        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1 and token_types.shape[0] == total_seq_len:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)

        # 6. Decoder layers - layer-by-layer with loading/unloading
        text_model = inner_model.language_model

        if use_flash_attn and _flash_attn_func is not None:
            hidden_states = self._run_decoder_flash_layer_by_layer(
                inputs_embeds, position_ids, token_types, text_model,
            )
        else:
            hidden_states = self._run_decoder_standard_layer_by_layer(
                inputs_embeds, position_ids, token_types, text_model,
            )

        # 7. Norm
        hidden_states = text_model.norm(hidden_states)

        # 8. Final layer -> pixel predictions
        x_pred = inner_model.final_layer2(hidden_states)

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            x_pred=x_pred,
        )

    def _run_decoder_flash_layer_by_layer(
        self, inputs_embeds, position_ids, token_types, text_model,
    ):
        device = self.device
        dtype = self.dtype

        # Position embeddings
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
        cos, sin = position_embeddings

        is_gen = token_types[0].bool()
        idx_ar = torch.nonzero(~is_gen, as_tuple=False).squeeze(-1)

        hidden_states = inputs_embeds
        head_dim = text_model.layers[0].self_attn.head_dim  # meta tensor, shape works

        for layer_idx in range(self.num_decoder_layers):
            sd = self._load_decoder_layer(layer_idx)
            decoder_layer = text_model.layers[layer_idx]

            hidden_states = self._flash_layer_forward(
                hidden_states, decoder_layer, cos, sin, idx_ar, head_dim,
            )

            self._unload_decoder_layer(sd)

        return hidden_states

    def _flash_layer_forward(
        self, hidden_states, decoder_layer, cos, sin, idx_ar, head_dim,
    ):
        attn = decoder_layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        # input_layernorm
        residual = hidden_states
        normed = decoder_layer.input_layernorm(hidden_states)

        # Q, K, V projections
        q = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        q_r = q.transpose(1, 2)
        k_r = k.transpose(1, 2)
        q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin)
        q = q_r.transpose(1, 2).contiguous()
        k = k_r.transpose(1, 2).contiguous()
        v = v.contiguous()

        softmax_scale = head_dim ** (-0.5)

        # Pass 1: causal on AR tokens
        q_ar = q[:, idx_ar].contiguous()
        k_ar = k[:, idx_ar].contiguous()
        v_ar = v[:, idx_ar].contiguous()
        result_ar = _flash_attn_func(
            q_ar.to(torch.bfloat16), k_ar.to(torch.bfloat16), v_ar.to(torch.bfloat16),
            softmax_scale=softmax_scale, causal=True,
        )
        out_ar = result_ar[0] if isinstance(result_ar, tuple) else result_ar

        # Pass 2: full bidirectional on all tokens
        result_full = _flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            softmax_scale=softmax_scale, causal=False,
        )
        out_full = result_full[0] if isinstance(result_full, tuple) else result_full

        # Replace AR positions with causal result
        out_full = out_full.clone()
        out_full[:, idx_ar] = out_ar

        # Output projection
        attn_output = out_full.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _run_decoder_standard_layer_by_layer(
        self, inputs_embeds, position_ids, token_types, text_model,
    ):
        device = self.device
        dtype = self.dtype
        batch_size, total_seq_len, hidden_size = inputs_embeds.shape

        # Position embeddings
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids_3d = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
            position_ids_3d = position_ids

        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids_3d)

        # Build 4D attention mask
        min_val = torch.finfo(dtype).min
        attn_masks = []
        for b in range(batch_size):
            causal = torch.full((total_seq_len, total_seq_len), min_val, device=device, dtype=dtype)
            causal = torch.triu(causal, diagonal=1)
            gen_positions = token_types[b].bool()
            causal[gen_positions, :] = 0
            attn_masks.append(causal)
        attention_mask_4d = torch.stack(attn_masks, dim=0).unsqueeze(1)

        hidden_states = inputs_embeds

        for layer_idx in range(self.num_decoder_layers):
            sd = self._load_decoder_layer(layer_idx)
            decoder_layer = text_model.layers[layer_idx]

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=text_position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )

            self._unload_decoder_layer(sd)

        return hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
    ) -> Qwen3VLCausalLMOutputWithPast:
        outputs = self._forward_generation(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=vinputs,
            timestep=timestep,
            token_types=token_types,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_flash_attn=use_flash_attn,
        )
        return Qwen3VLCausalLMOutputWithPast(
            x_pred=outputs.x_pred,
        )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)
