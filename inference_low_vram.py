import argparse
import torch
import os
from transformers import AutoProcessor
from models.pipeline import generate_image, DEFAULT_TIMESTEPS
from models.low_vram_model import LowVRAMImageGenerationModel


def add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def main():
    p = argparse.ArgumentParser("Low-VRAM HiDream-O1 inference (AirLLM-style layer-by-layer)")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/HiDream-O1-Image-Dev",
                   help="Path to huggingface model")
    p.add_argument("--prompt", type=str,
                   default="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.")
    p.add_argument("--ref_images", nargs="*", default=[], help="Path to reference images.")
    p.add_argument("--output_image", type=str, default="output.png")
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--model_type", type=str, default="full", choices=["full", "dev"])
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--noise_scale_start", type=float, default=7.5)
    p.add_argument("--noise_scale_end", type=float, default=7.5)
    p.add_argument("--noise_clip_std", type=float, default=2.5)
    p.add_argument("--use_flash_attn", action="store_true", default=True,
                   help="Use flash attention two-pass (faster, less memory)")
    p.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false",
                   help="Disable flash attention (use 4D mask)")
    p.add_argument("--layer_shards_path", type=str, default=None,
                   help="Path to save/load splitted layer files (default: inside model_path)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Model dtype")
    p.add_argument(
        "--keep_original_aspect",
        action="store_true",
        help="When exactly one reference image is provided, resize it with "
             "max_size=2048 and use its dimensions for the target image "
             "(preserves the reference's aspect ratio).",
    )
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for inference."

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[inference] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"[inference] Loading model with AirLLM-style layer-by-layer loading (dtype={dtype})...")
    model = LowVRAMImageGenerationModel(
        model_path=args.model_path,
        device="cuda:0",
        dtype=dtype,
        layer_shards_path=args.layer_shards_path,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_image)), exist_ok=True)
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    extra_kwargs = {}
    if args.model_type == "full":
        num_inference_steps = 50
        guidance_scale = args.guidance_scale
        shift = 3.0
        timesteps_list = None
        scheduler_name = "default"
    else:
        num_inference_steps = 28
        guidance_scale = 0.0
        shift = 1.0
        timesteps_list = DEFAULT_TIMESTEPS
        scheduler_name = "flash"
        extra_kwargs["noise_scale_start"] = args.noise_scale_start
        extra_kwargs["noise_scale_end"] = args.noise_scale_end
        extra_kwargs["noise_clip_std"] = args.noise_clip_std

    # Inject the user's use_flash_attn preference into model calls.
    # pipeline.py hardcodes use_flash_attn=True; we override with user choice.
    _orig_forward = model.forward

    def _patched_forward(**kwargs):
        kwargs["use_flash_attn"] = args.use_flash_attn
        kwargs.pop("precomputed_image_embeds", None)
        kwargs.pop("precomputed_deepstack_image_embeds", None)
        return _orig_forward(**kwargs)

    model.forward = _patched_forward
    model.__call__ = _patched_forward

    image = generate_image(
        model=model,
        processor=processor,
        prompt=args.prompt,
        ref_image_paths=args.ref_images,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        timesteps_list=timesteps_list,
        scheduler_name=scheduler_name,
        seed=args.seed,
        keep_original_aspect=args.keep_original_aspect,
        **extra_kwargs,
    )
    image.save(args.output_image)
    print(f"[inference] Saved -> {args.output_image}")


if __name__ == "__main__":
    main()
