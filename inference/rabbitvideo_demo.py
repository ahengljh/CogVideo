"""
RabbitVideo Demo and Comparison Script for CogVideoX

This script demonstrates how to use RabbitVideo and compare it with other memory optimization methods:
1. Standard mode (no optimization) - baseline
2. enable_sequential_cpu_offload() - HuggingFace's built-in method
3. enable_rabbitvideo() - Our RabbitVideo implementation

Usage:
    # Test with RabbitVideo
    python rabbitvideo_demo.py --method rabbitvideo --max_gpu_blocks 5

    # Test with sequential CPU offload
    python rabbitvideo_demo.py --method sequential

    # Test standard mode (requires large GPU)
    python rabbitvideo_demo.py --method standard

    # Compare all methods
    python rabbitvideo_demo.py --method all --num_inference_steps 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import torch

# Add rabbitvideo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from rabbitvideo import enable_rabbitvideo


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryBenchmark:
    """Track memory usage and timing for benchmarking."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.start_memory = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / (1024**3)

        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        logger.info(f"Completed {self.name}")
        logger.info(f"  Duration: {duration:.2f}s")
        if torch.cuda.is_available():
            logger.info(f"  Peak Memory: {self.peak_memory:.2f}GB")
            logger.info(f"  Memory Delta: {self.peak_memory - self.start_memory:.2f}GB")

    def get_stats(self):
        return {
            "name": self.name,
            "duration_s": self.end_time - self.start_time if self.end_time else 0,
            "peak_memory_gb": self.peak_memory,
            "memory_delta_gb": self.peak_memory - self.start_memory,
        }


def generate_with_standard(
    model_path: str,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
    dtype: torch.dtype,
    seed: int,
    output_path: str,
):
    """Generate video with standard mode (no offloading)."""
    logger.info("=" * 70)
    logger.info("Method: STANDARD (No Offloading)")
    logger.info("=" * 70)

    with MemoryBenchmark("Standard Mode") as bench:
        # Load pipeline
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        # Move to GPU
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # Generate
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        export_to_video(video, output_path, fps=16)

    return bench.get_stats()


def generate_with_sequential_offload(
    model_path: str,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
    dtype: torch.dtype,
    seed: int,
    output_path: str,
):
    """Generate video with sequential CPU offload."""
    logger.info("=" * 70)
    logger.info("Method: SEQUENTIAL CPU OFFLOAD (HuggingFace)")
    logger.info("=" * 70)

    with MemoryBenchmark("Sequential CPU Offload") as bench:
        # Load pipeline
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        # Enable sequential CPU offload
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # Generate
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        export_to_video(video, output_path, fps=16)

    return bench.get_stats()


def generate_with_rabbitvideo(
    model_path: str,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
    dtype: torch.dtype,
    seed: int,
    output_path: str,
    max_gpu_blocks: int = 5,
    enable_kv_cache: bool = False,
):
    """Generate video with RabbitVideo."""
    logger.info("=" * 70)
    logger.info("Method: RABBITVIDEO")
    logger.info("=" * 70)

    with MemoryBenchmark("RabbitVideo") as bench:
        # Load pipeline
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        # Move to GPU first (RabbitVideo will manage offloading)
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # Enable RabbitVideo
        enable_rabbitvideo(
            pipe,
            max_gpu_blocks=max_gpu_blocks,
            initial_gpu_blocks=max_gpu_blocks,
            enable_kv_cache=enable_kv_cache,
            offload_text_encoder=True,
            manage_vae=True,
            verbose=True,
        )

        # Generate
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        export_to_video(video, output_path, fps=16)

        # Print RabbitVideo stats
        pipe.print_rabbitvideo_stats()

    return bench.get_stats()


def print_comparison_table(results: list):
    """Print comparison table of results."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)

    # Print header
    print(f"{'Method':<25} {'Duration (s)':<15} {'Peak Mem (GB)':<15} {'Mem Delta (GB)':<15}")
    print("-" * 70)

    # Print each result
    for result in results:
        print(f"{result['name']:<25} {result['duration_s']:<15.2f} "
              f"{result['peak_memory_gb']:<15.2f} {result['memory_delta_gb']:<15.2f}")

    # Calculate improvements vs baseline
    if len(results) > 1 and results[0]['name'] == 'Sequential CPU Offload':
        baseline = results[0]
        print("\n" + "=" * 70)
        print("IMPROVEMENT vs Sequential CPU Offload")
        print("=" * 70)

        for result in results[1:]:
            time_overhead = ((result['duration_s'] - baseline['duration_s']) / baseline['duration_s']) * 100
            mem_reduction = ((baseline['peak_memory_gb'] - result['peak_memory_gb']) / baseline['peak_memory_gb']) * 100

            print(f"\n{result['name']}:")
            print(f"  Time Overhead: {time_overhead:+.1f}%")
            print(f"  Memory Reduction: {mem_reduction:+.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="RabbitVideo Demo and Comparison for CogVideoX"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rabbitvideo",
        choices=["standard", "sequential", "rabbitvideo", "all"],
        help="Memory optimization method to use"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Model path (use 2b for testing, 5b for production)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A girl riding a bike through a beautiful park.",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames to generate (49 for CogVideoX 1.0)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps (lower for faster testing)"
    )
    parser.add_argument(
        "--max_gpu_blocks",
        type=int,
        default=5,
        help="Max GPU blocks for RabbitVideo (5 for 24GB, 3 for 12GB)"
    )
    parser.add_argument(
        "--enable_kv_cache",
        action="store_true",
        help="Enable experimental KV cache optimization"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for computation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for videos"
    )

    args = parser.parse_args()

    # Setup
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. RabbitVideo requires a CUDA-enabled GPU.")
        return

    logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")

    # Run tests
    results = []

    if args.method == "standard" or args.method == "all":
        try:
            stats = generate_with_standard(
                args.model_path, args.prompt, args.num_frames,
                args.num_inference_steps, dtype, args.seed,
                str(output_dir / "output_standard.mp4")
            )
            results.append(stats)
        except RuntimeError as e:
            logger.error(f"Standard mode failed (likely OOM): {e}")

    if args.method == "sequential" or args.method == "all":
        stats = generate_with_sequential_offload(
            args.model_path, args.prompt, args.num_frames,
            args.num_inference_steps, dtype, args.seed,
            str(output_dir / "output_sequential.mp4")
        )
        results.append(stats)

    if args.method == "rabbitvideo" or args.method == "all":
        stats = generate_with_rabbitvideo(
            args.model_path, args.prompt, args.num_frames,
            args.num_inference_steps, dtype, args.seed,
            str(output_dir / "output_rabbitvideo.mp4"),
            max_gpu_blocks=args.max_gpu_blocks,
            enable_kv_cache=args.enable_kv_cache,
        )
        results.append(stats)

    # Print comparison
    if len(results) > 1:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
