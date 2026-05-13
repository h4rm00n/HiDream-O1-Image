# HiDream-O1-Image 推理指南

本项目提供两套推理脚本，分别面向不同显存配置的 GPU。

---

## 环境要求

- Python 3.10+
- CUDA GPU（至少 8 GB 显存，推荐 24 GB 以上）
- 依赖安装：`pip install -r requirements.txt`

---

## 1. 标准推理 — `inference.py`

适用于 **显存充足**（≥ 20 GB）的 GPU。所有模型权重一次性加载到显存，推理速度最快。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model_path` | str | `/root/converted_models/HiDream-O1-Image` | 模型路径（HF repo 或本地目录） |
| `--prompt` | str | （见脚本内） | 生成图像的描述文本 |
| `--ref_images` | list | `[]` | 参考图像路径列表，可用于图像条件生成 |
| `--output_image` | str | `output.png` | 输出图像保存路径 |
| `--height` | int | `2048` | 目标图像高度（会被 snap 到预定义分辨率） |
| `--width` | int | `2048` | 目标图像宽度（会被 snap 到预定义分辨率） |
| `--model_type` | str | `full` | 推理模式：`full`（50 步，高质量）或 `dev`（28 步，快速） |
| `--seed` | int | `32` | 随机种子 |
| `--guidance_scale` | float | `5.0` | 无分类器引导强度（`full` 模式有效） |
| `--noise_scale_start` | float | `7.5` | 首步噪声缩放（`dev` 模式） |
| `--noise_scale_end` | float | `7.5` | 末步噪声缩放（`dev` 模式） |
| `--noise_clip_std` | float | `2.5` | 噪声裁剪标准差（`dev` 模式） |
| `--keep_original_aspect` | flag | `False` | 配合单张参考图，用参考图的宽高比替代预定义分辨率 |

### 用法示例

```bash
# 文生图（全质量模式）
python inference.py --prompt "A cat sitting on a chair, oil painting" --output_image cat.png

# 快速模式（dev, 28 步）
python inference.py --model_type dev --prompt "A sunset beach" --output_image beach.png

# 参考图生图 + 保持原图比例
python inference.py \
    --prompt "A portrait in the same style" \
    --ref_images ref.jpg \
    --keep_original_aspect \
    --output_image portrait.png
```

---

## 2. 低显存推理 — `inference_low_vram.py`

借鉴 [AirLLM](https://github.com/lyogavin/airllm) 的逐层加载机制，**只为计算当前层加载权重到显存，计算完成后立即释放**。适合 8–12 GB 显存的 GPU。

> **首次运行**会自动将模型权重拆分为逐层 safetensors 文件（存储在 `model_path/splitted_layers/`），该过程约需 30–60 秒。后续推理直接复用已拆分的文件。

### 额外参数（相比标准版新增）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--use_flash_attn` | flag | `True` | 启用 Flash Attention 双通道 | 
| `--no_flash_attn` | flag | — | 禁用 Flash Attention，回退到 4D mask 标准路径 |
| `--layer_shards_path` | str | `None` | 层级权重保存目录（默认在 `model_path` 内） |
| `--dtype` | str | `bfloat16` | 模型数据类型：`bfloat16` / `float16` / `float32` |

其余参数与标准版一致。

### 用法示例

```bash
# 文生图（dev 模式，低显存）
python inference_low_vram.py \
    --model_type dev \
    --prompt "A mountain landscape" \
    --output_image mountain.png

# 禁用 Flash Attention（如未安装 flash-attn）
python inference_low_vram.py --model_type dev --no_flash_attn --prompt "..." --output_image out.png

# 指定拆分文件存放路径
python inference_low_vram.py \
    --model_path /data/models/HiDream-O1-Image \
    --layer_shards_path /data/splitted_layers \
    --prompt "..." --output_image out.png

# 使用 float16 进一步节省显存
python inference_low_vram.py --dtype float16 --prompt "..." --output_image out.png
```

---

## 3. 预定义分辨率

模型在以下 11 个分辨率桶上训练。传入的 `--width` 和 `--height` 会自动匹配到最近的分辨率：

| 纵横比 | 分辨率 |
|---|---|
| 1:1 | 2048 × 2048 |
| 4:3 | 2304 × 1728 / 1728 × 2304 / 2496 × 1664 / 1664 × 2496 |
| 16:9 | 2560 × 1440 / 1440 × 2560 |
| ≈ 2.37:1 | 3104 × 1312 / 1312 × 3104 |
| ≈ 1.28:1 | 2304 × 1792 / 1792 × 2304 |

> 如需自定义分辨率，可使用 `--keep_original_aspect` + 一张参考图来绕过分辨率匹配（见 `models/utils.py:76` `get_rope_index_fix_point`）。

---

## 4. 推理模式对比

| 模式 | 步数 | guidance_scale | shift | scheduler | 说明 |
|---|---|---|---|---|---|
| `full` | 50 | 5.0 | 3.0 | UniPC | 高质量，细节丰富 |
| `dev` | 28 | 0.0 | 1.0 | Flash Euler | 快速，无 CFG 引导 |

---

## 5. 性能对比（2048×2048, dev 模式, bf16）

| | 标准版 `inference.py` | 低显存版 `inference_low_vram.py` |
|---|---|---|
| 加载时间 | ~5 秒 | ~40 秒（首次）/ ~5 秒（后续） |
| 推理速度 | ~0.9 s/step | ~9.3 s/step |
| 28 步总耗时 | ~26 秒 | ~4 分 19 秒 |
| 显存占用 | ~15–20 GB | ~3–5 GB |
| 适用 GPU | RTX 4090 / A100 | RTX 3070 / 4060 Ti |

> 低显存版慢约 10 倍——本质是以速度换显存。每个去噪步需遍历 35 个 transformer 层，逐层从磁盘加载→传输到 GPU→计算→释放。

---

## 6. 架构说明

```
Qwen3VLForConditionalGeneration
├── model (Qwen3VLModel)
│   ├── visual (Qwen3VLVisionModel)         ← 视觉编码器（仅参考图模式加载）
│   ├── language_model (Qwen3VLTextModel)
│   │   ├── embed_tokens + rotary_emb       ← 常驻 GPU（~1.2 GB）
│   │   ├── layers.0 ... layers.34          ← 逐层加载/释放（每层 ~300 MB）
│   │   └── norm                            ← 常驻 GPU
│   ├── t_embedder1                          ← 常驻 GPU
│   ├── x_embedder                           ← 常驻 GPU
│   └── final_layer2                         ← 常驻 GPU
└── lm_head                                  ← 常驻 GPU（与 embed_tokens 权重绑定）
```

低显存版仅在 GPU 保留**当前正在计算的 1 个 decoder 层**，其余层权重存储在磁盘的 `splitted_layers/` 目录中。
