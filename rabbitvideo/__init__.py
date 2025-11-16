"""
RabbitVideo: Memory optimization system for video diffusion models

RabbitVideo enables running large video diffusion models (like CogVideoX) on consumer GPUs
by reducing peak memory usage through intelligent block-level offloading.

Versions:
- v1.0: Basic block-level offloading with LRU eviction
- v2.0: Enhanced with async prefetching, smart pinning, and batched transfers (RECOMMENDED)

Usage (v2.0 - Recommended):
    from rabbitvideo import enable_rabbitvideo_v2

    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    enable_rabbitvideo_v2(pipe, max_gpu_blocks=5)

    # Generate video as usual
    video = pipe(prompt="A girl riding a bike.").frames[0]

Usage (v1.0 - Legacy):
    from rabbitvideo import enable_rabbitvideo

    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    enable_rabbitvideo(pipe, max_gpu_blocks=5)
"""

from .core import MemoryMonitor, BlockTracker, BlockManager, RabbitVideoOffloader
from .pipeline_integration import enable_rabbitvideo

# V2 components
from .core_v2 import RabbitVideoOffloaderV2
from .pipeline_integration_v2 import enable_rabbitvideo_v2
from .advanced import (
    SmartBlockPinner,
    AsyncBlockPrefetcher,
    BatchedBlockTransfer,
    AdaptiveBlockManagerV2,
)

__version__ = "2.0.0"
__all__ = [
    # Core v1
    "MemoryMonitor",
    "BlockTracker",
    "BlockManager",
    "RabbitVideoOffloader",
    "enable_rabbitvideo",
    # Core v2 (recommended)
    "RabbitVideoOffloaderV2",
    "enable_rabbitvideo_v2",
    "SmartBlockPinner",
    "AsyncBlockPrefetcher",
    "BatchedBlockTransfer",
    "AdaptiveBlockManagerV2",
]
