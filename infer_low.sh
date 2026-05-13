export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
python inference_low_vram.py \
    --model_path /home/harmoon/projects/models/HiDream-O1-Image-Dev  \
    --prompt "A mountain landscape" \
    --output_image out.png \
    --model_type dev \
    --no_flash_attn \
    --width 2048 \
    --height 2048
