"""
Pipeline integration for RabbitVideo with CogVideoX.

This module provides easy-to-use integration with HuggingFace diffusers pipelines,
adding the enable_rabbitvideo() method to CogVideoX pipelines.
"""

import logging
from typing import Optional

import torch

from .core import RabbitVideoOffloader


logger = logging.getLogger(__name__)


def enable_rabbitvideo(
    pipeline,
    max_gpu_blocks: int = 5,
    initial_gpu_blocks: int = 5,
    enable_kv_cache: bool = False,
    offload_text_encoder: bool = True,
    manage_vae: bool = True,
    verbose: bool = True,
):
    """
    Enable RabbitVideo memory optimization on a CogVideoX pipeline.

    This function monkey-patches the pipeline to add intelligent memory management
    that reduces peak memory usage by ~60% with only ~15% time overhead.

    Args:
        pipeline: CogVideoX pipeline instance (CogVideoXPipeline, CogVideoXImageToVideoPipeline, etc.)
        max_gpu_blocks: Maximum transformer blocks to keep on GPU during inference (default: 5)
        initial_gpu_blocks: Number of blocks to keep on GPU initially (default: 5)
        enable_kv_cache: Enable experimental KV cache optimization for ~5-10% speedup (default: False)
        offload_text_encoder: Offload text encoder to CPU after encoding (default: True)
        manage_vae: Automatically manage VAE transfers during inference (default: True)
        verbose: Enable verbose logging (default: True)

    Returns:
        The modified pipeline with RabbitVideo enabled

    Example:
        >>> from diffusers import CogVideoXPipeline
        >>> import torch
        >>> from rabbitvideo import enable_rabbitvideo
        >>>
        >>> pipe = CogVideoXPipeline.from_pretrained(
        ...     "THUDM/CogVideoX-5b",
        ...     torch_dtype=torch.bfloat16
        ... )
        >>>
        >>> # Enable RabbitVideo instead of sequential_cpu_offload
        >>> enable_rabbitvideo(pipe, max_gpu_blocks=5)
        >>>
        >>> # Generate video as usual
        >>> video = pipe(prompt="A girl riding a bike.").frames[0]

    Notes:
        - RabbitVideo is compatible with LoRA, but may need adjustment for other techniques
        - For 24GB GPUs, max_gpu_blocks=5 is recommended
        - For 12GB GPUs, try max_gpu_blocks=3
        - enable_kv_cache is experimental (v1.1 feature)
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("Enabling RabbitVideo Memory Optimization")
        logger.info("=" * 70)

    # Get device from pipeline
    if hasattr(pipeline, 'device'):
        device = pipeline.device
    elif hasattr(pipeline, '_execution_device'):
        device = pipeline._execution_device
    else:
        device = torch.device('cuda:0')

    # Get models from pipeline
    transformer = getattr(pipeline, 'transformer', None)
    vae = getattr(pipeline, 'vae', None) if manage_vae else None
    text_encoder = getattr(pipeline, 'text_encoder', None) if offload_text_encoder else None
    text_encoder_2 = getattr(pipeline, 'text_encoder_2', None) if offload_text_encoder else None

    if transformer is None:
        raise ValueError("Pipeline does not have a 'transformer' attribute. "
                        "RabbitVideo currently only supports transformer-based models.")

    # Create RabbitVideo offloader
    offloader = RabbitVideoOffloader(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        max_gpu_blocks=max_gpu_blocks,
        initial_gpu_blocks=initial_gpu_blocks,
        gpu_device=device,
        enable_kv_cache=enable_kv_cache,
        verbose=verbose,
    )

    # Store offloader in pipeline
    pipeline._rabbitvideo_offloader = offloader

    # Monkey-patch the pipeline's __call__ method to manage VAE transfers
    if manage_vae and vae is not None:
        original_call = pipeline.__call__

        def rabbitvideo_call(*args, **kwargs):
            """Wrapped __call__ that manages VAE transfers."""
            # Offload VAE before transformer inference
            offloader.offload_vae()

            # Run original inference (transformer will use dynamic block loading)
            try:
                result = original_call(*args, **kwargs)
            finally:
                # Load VAE back for decoding
                offloader.load_vae()

            return result

        pipeline.__call__ = rabbitvideo_call

    # Monkey-patch text encoding if requested
    if offload_text_encoder and (text_encoder is not None or text_encoder_2 is not None):
        # The encoding typically happens in encode_prompt method
        if hasattr(pipeline, 'encode_prompt'):
            original_encode = pipeline.encode_prompt

            def rabbitvideo_encode_prompt(*args, **kwargs):
                """Wrapped encode_prompt that offloads encoders after use."""
                result = original_encode(*args, **kwargs)
                offloader.offload_text_encoders()
                return result

            pipeline.encode_prompt = rabbitvideo_encode_prompt

    # Add helper methods to pipeline
    def get_rabbitvideo_stats():
        """Get RabbitVideo statistics."""
        return offloader.get_stats()

    def print_rabbitvideo_stats():
        """Print RabbitVideo statistics."""
        offloader.print_stats()

    def disable_rabbitvideo():
        """Disable RabbitVideo and restore original behavior."""
        logger.info("Disabling RabbitVideo")
        # This is a placeholder - full restoration would require more complex logic
        logger.warning("disable_rabbitvideo is not fully implemented. "
                      "Please recreate the pipeline to disable RabbitVideo.")

    pipeline.get_rabbitvideo_stats = get_rabbitvideo_stats
    pipeline.print_rabbitvideo_stats = print_rabbitvideo_stats
    pipeline.disable_rabbitvideo = disable_rabbitvideo

    if verbose:
        logger.info("RabbitVideo enabled successfully!")
        logger.info(f"  Max GPU blocks: {max_gpu_blocks}")
        logger.info(f"  Initial GPU blocks: {initial_gpu_blocks}")
        logger.info(f"  KV cache: {'enabled' if enable_kv_cache else 'disabled'}")
        logger.info(f"  Text encoder management: {'enabled' if offload_text_encoder else 'disabled'}")
        logger.info(f"  VAE management: {'enabled' if manage_vae else 'disabled'}")
        logger.info("=" * 70)

    return pipeline


def patch_cogvideox_pipeline():
    """
    Monkey-patch CogVideoX pipeline classes to add enable_rabbitvideo() method.

    This allows using pipe.enable_rabbitvideo() similar to pipe.enable_sequential_cpu_offload().

    Example:
        >>> from rabbitvideo.pipeline_integration import patch_cogvideox_pipeline
        >>> patch_cogvideox_pipeline()
        >>>
        >>> from diffusers import CogVideoXPipeline
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
        >>> pipe.enable_rabbitvideo(max_gpu_blocks=5)
    """
    try:
        from diffusers import (
            CogVideoXPipeline,
            CogVideoXImageToVideoPipeline,
            CogVideoXVideoToVideoPipeline,
        )

        def make_enable_method():
            def enable_rabbitvideo_method(
                self,
                max_gpu_blocks: int = 5,
                initial_gpu_blocks: int = 5,
                enable_kv_cache: bool = False,
                offload_text_encoder: bool = True,
                manage_vae: bool = True,
                verbose: bool = True,
            ):
                return enable_rabbitvideo(
                    self,
                    max_gpu_blocks=max_gpu_blocks,
                    initial_gpu_blocks=initial_gpu_blocks,
                    enable_kv_cache=enable_kv_cache,
                    offload_text_encoder=offload_text_encoder,
                    manage_vae=manage_vae,
                    verbose=verbose,
                )
            return enable_rabbitvideo_method

        # Add method to all pipeline classes
        for pipeline_class in [CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXVideoToVideoPipeline]:
            pipeline_class.enable_rabbitvideo = make_enable_method()

        logger.info("Successfully patched CogVideoX pipeline classes with enable_rabbitvideo()")

    except ImportError as e:
        logger.warning(f"Could not patch CogVideoX pipelines: {e}")
        logger.warning("diffusers library may not be installed or CogVideoX pipelines not available")
