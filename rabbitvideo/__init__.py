"""
RabbitVideo: Memory optimization system for video diffusion models

RabbitVideo enables running large video diffusion models (like CogVideoX) on consumer GPUs
by reducing peak memory usage through intelligent block-level offloading.

Usage:
    from rabbitvideo import enable_rabbitvideo

    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    enable_rabbitvideo(pipe)

    # Generate video as usual
    video = pipe(prompt="A girl riding a bike.").frames[0]
"""

from .core import MemoryMonitor, BlockTracker, BlockManager, RabbitVideoOffloader
from .pipeline_integration import enable_rabbitvideo

__version__ = "1.1.0"
__all__ = [
    "MemoryMonitor",
    "BlockTracker",
    "BlockManager",
    "RabbitVideoOffloader",
    "enable_rabbitvideo",
]
