"""
Pipeline integration for RabbitVideo v2.0

Provides enable_rabbitvideo_v2() for easy integration with CogVideoX pipelines.
"""

import logging
from typing import Optional

import torch

from .core_v2 import RabbitVideoOffloaderV2


logger = logging.getLogger(__name__)


def enable_rabbitvideo_v2(
    pipeline,
    max_gpu_blocks: int = 5,
    initial_gpu_blocks: int = 5,
    enable_kv_cache: bool = False,
    enable_prefetch: bool = True,
    enable_smart_pinning: bool = True,
    enable_batching: bool = True,
    offload_text_encoder: bool = True,
    manage_vae: bool = True,
    verbose: bool = True,
):
    """
    Enable RabbitVideo v2.0 memory optimization on a CogVideoX pipeline.

    V2.0 improvements over v1.0:
    - Async prefetching: Overlaps transfers with compute (hides latency)
    - Smart block pinning: Learns access patterns, pins hot blocks
    - Batched transfers: Reduces overhead from setup/teardown
    - Expected: Faster than sequential CPU offload with better memory

    Args:
        pipeline: CogVideoX pipeline instance
        max_gpu_blocks: Max transformer blocks on GPU (default: 5)
        initial_gpu_blocks: Initial blocks on GPU (default: 5)
        enable_kv_cache: Enable KV cache optimization (default: False)
        enable_prefetch: Enable async prefetching - NEW in v2 (default: True)
        enable_smart_pinning: Enable smart block pinning - NEW in v2 (default: True)
        enable_batching: Enable batched transfers - NEW in v2 (default: True)
        offload_text_encoder: Offload text encoder after use (default: True)
        manage_vae: Auto-manage VAE transfers (default: True)
        verbose: Enable verbose logging (default: True)

    Returns:
        The modified pipeline with RabbitVideo v2 enabled

    Example:
        >>> from diffusers import CogVideoXPipeline
        >>> import torch
        >>> from rabbitvideo.pipeline_integration_v2 import enable_rabbitvideo_v2
        >>>
        >>> pipe = CogVideoXPipeline.from_pretrained(
        ...     "THUDM/CogVideoX-5b",
        ...     torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>>
        >>> # Enable v2 with all optimizations
        >>> enable_rabbitvideo_v2(pipe, max_gpu_blocks=5)
        >>>
        >>> # Or disable specific v2 features
        >>> enable_rabbitvideo_v2(
        ...     pipe,
        ...     enable_prefetch=False,  # Disable prefetching
        ...     enable_smart_pinning=True,  # Keep smart pinning
        ... )
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("Enabling RabbitVideo v2.0 Memory Optimization")
        logger.info("=" * 70)

    # Get device
    if hasattr(pipeline, 'device'):
        device = pipeline.device
    elif hasattr(pipeline, '_execution_device'):
        device = pipeline._execution_device
    else:
        device = torch.device('cuda:0')

    # Get models
    transformer = getattr(pipeline, 'transformer', None)
    vae = getattr(pipeline, 'vae', None) if manage_vae else None
    text_encoder = getattr(pipeline, 'text_encoder', None) if offload_text_encoder else None
    text_encoder_2 = getattr(pipeline, 'text_encoder_2', None) if offload_text_encoder else None

    if transformer is None:
        raise ValueError("Pipeline does not have a 'transformer' attribute.")

    # Create v2 offloader
    offloader = RabbitVideoOffloaderV2(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        max_gpu_blocks=max_gpu_blocks,
        initial_gpu_blocks=initial_gpu_blocks,
        gpu_device=device,
        enable_kv_cache=enable_kv_cache,
        enable_prefetch=enable_prefetch,
        enable_smart_pinning=enable_smart_pinning,
        enable_batching=enable_batching,
        verbose=verbose,
    )

    # Store offloader
    pipeline._rabbitvideo_offloader = offloader
    pipeline._rabbitvideo_version = "2.0"

    # Wrap pipeline.__call__ for VAE management and step tracking
    if manage_vae and vae is not None:
        original_call = pipeline.__call__

        def rabbitvideo_v2_call(*args, **kwargs):
            """Wrapped __call__ with VAE management and step tracking."""
            # Offload VAE before transformer
            offloader.offload_vae()

            # Wrap the denoising loop to track steps for smart pinning
            # Note: This is a simplified version. In practice, we'd need to hook
            # into the actual denoising loop, but that requires pipeline-specific logic.

            try:
                result = original_call(*args, **kwargs)
            finally:
                # Load VAE back
                offloader.load_vae()

            return result

        pipeline.__call__ = rabbitvideo_v2_call

    # Text encoder offloading
    if offload_text_encoder and (text_encoder is not None or text_encoder_2 is not None):
        if hasattr(pipeline, 'encode_prompt'):
            original_encode = pipeline.encode_prompt

            def rabbitvideo_v2_encode_prompt(*args, **kwargs):
                """Wrapped encode_prompt."""
                result = original_encode(*args, **kwargs)
                offloader.offload_text_encoders()
                return result

            pipeline.encode_prompt = rabbitvideo_v2_encode_prompt

    # Add helper methods
    def get_rabbitvideo_stats():
        """Get RabbitVideo v2 statistics."""
        return offloader.get_stats()

    def print_rabbitvideo_stats():
        """Print RabbitVideo v2 statistics."""
        offloader.print_stats()

    pipeline.get_rabbitvideo_stats = get_rabbitvideo_stats
    pipeline.print_rabbitvideo_stats = print_rabbitvideo_stats

    if verbose:
        logger.info("RabbitVideo v2.0 enabled successfully!")
        logger.info(f"  Version: 2.0")
        logger.info(f"  Max GPU blocks: {max_gpu_blocks}")
        logger.info(f"  V2 Features:")
        logger.info(f"    - Async Prefetching: {enable_prefetch}")
        logger.info(f"    - Smart Block Pinning: {enable_smart_pinning}")
        logger.info(f"    - Batched Transfers: {enable_batching}")
        logger.info(f"  V1 Features:")
        logger.info(f"    - KV Cache: {enable_kv_cache}")
        logger.info(f"    - Text Encoder Management: {offload_text_encoder}")
        logger.info(f"    - VAE Management: {manage_vae}")
        logger.info("=" * 70)

    return pipeline
