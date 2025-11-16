"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

# Add rabbitvideo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rabbitvideo import enable_rabbitvideo, enable_rabbitvideo_v2
    RABBITVIDEO_AVAILABLE = True
    RABBITVIDEO_V2_AVAILABLE = True
except ImportError:
    RABBITVIDEO_AVAILABLE = False
    RABBITVIDEO_V2_AVAILABLE = False
    logging.warning("RabbitVideo not available. Install to use --use_rabbitvideo flag.")


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    use_rabbitvideo: bool = False,
    rabbitvideo_version: str = "v2",
    max_gpu_blocks: int = 5,
    enable_kv_cache: bool = False,
    enable_prefetch: bool = True,
    enable_smart_pinning: bool = True,
    enable_batching: bool = True,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    - use_rabbitvideo (bool): Use RabbitVideo memory optimization instead of sequential CPU offload.
    - rabbitvideo_version (str): RabbitVideo version: 'v1' or 'v2' (default: 'v2').
    - max_gpu_blocks (int): Maximum transformer blocks to keep on GPU (for RabbitVideo, default: 5).
    - enable_kv_cache (bool): Enable experimental KV cache optimization (for RabbitVideo, default: False).
    - enable_prefetch (bool): Enable async prefetching (RabbitVideo v2 only, default: True).
    - enable_smart_pinning (bool): Enable smart block pinning (RabbitVideo v2 only, default: True).
    - enable_batching (bool): Enable batched transfers (RabbitVideo v2 only, default: True).
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # You can use RabbitVideo for better memory efficiency:
    # V2 (recommended): python cli_demo.py --prompt "..." --use_rabbitvideo --rabbitvideo_version v2
    # V1 (legacy): python cli_demo.py --prompt "..." --use_rabbitvideo --rabbitvideo_version v1

    if use_rabbitvideo:
        if not RABBITVIDEO_AVAILABLE:
            raise ImportError(
                "RabbitVideo is not available. Please check that rabbitvideo/ directory exists."
            )

        pipe.to("cuda")

        if rabbitvideo_version == "v2":
            if not RABBITVIDEO_V2_AVAILABLE:
                raise ImportError("RabbitVideo v2 not available. Use --rabbitvideo_version v1")

            logging.info("Using RabbitVideo v2.0 memory optimization")
            enable_rabbitvideo_v2(
                pipe,
                max_gpu_blocks=max_gpu_blocks,
                initial_gpu_blocks=max_gpu_blocks,
                enable_kv_cache=enable_kv_cache,
                enable_prefetch=enable_prefetch,
                enable_smart_pinning=enable_smart_pinning,
                enable_batching=enable_batching,
                offload_text_encoder=True,
                manage_vae=True,
                verbose=True,
            )
        else:  # v1
            logging.info("Using RabbitVideo v1.0 memory optimization")
            enable_rabbitvideo(
                pipe,
                max_gpu_blocks=max_gpu_blocks,
                initial_gpu_blocks=max_gpu_blocks,
                enable_kv_cache=enable_kv_cache,
                offload_text_encoder=True,
                manage_vae=True,
                verbose=True,
            )
    else:
        # Default: use sequential CPU offload
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    )
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path save generated video"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of steps for the inference process"
    )
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument(
        "--height", type=int, default=None, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="The frames per second for the generated video"
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument(
        "--use_rabbitvideo",
        action="store_true",
        help="Use RabbitVideo memory optimization (better than sequential CPU offload)",
    )
    parser.add_argument(
        "--rabbitvideo_version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="RabbitVideo version: v1 (legacy) or v2 (recommended, faster)",
    )
    parser.add_argument(
        "--max_gpu_blocks",
        type=int,
        default=5,
        help="Max transformer blocks on GPU for RabbitVideo (5 for 24GB, 3 for 12GB)",
    )
    parser.add_argument(
        "--enable_kv_cache",
        action="store_true",
        help="Enable experimental KV cache optimization for RabbitVideo",
    )
    parser.add_argument(
        "--disable_prefetch",
        action="store_true",
        help="Disable async prefetching (RabbitVideo v2 only)",
    )
    parser.add_argument(
        "--disable_smart_pinning",
        action="store_true",
        help="Disable smart block pinning (RabbitVideo v2 only)",
    )
    parser.add_argument(
        "--disable_batching",
        action="store_true",
        help="Disable batched transfers (RabbitVideo v2 only)",
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        use_rabbitvideo=args.use_rabbitvideo,
        rabbitvideo_version=args.rabbitvideo_version,
        max_gpu_blocks=args.max_gpu_blocks,
        enable_kv_cache=args.enable_kv_cache,
        enable_prefetch=not args.disable_prefetch,
        enable_smart_pinning=not args.disable_smart_pinning,
        enable_batching=not args.disable_batching,
    )
