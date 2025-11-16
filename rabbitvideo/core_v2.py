"""
RabbitVideo v2.0 - Enhanced offloader with performance optimizations.

Improvements over v1.0:
- Async prefetching to overlap transfers with compute
- Smart block pinning based on learned access patterns
- Batched transfers to reduce overhead
- Adaptive management for better performance

Expected results: Faster than sequential CPU offload with better memory efficiency.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .core import MemoryMonitor, BlockTracker, RabbitVideoOffloader
from .advanced import AdaptiveBlockManagerV2


logger = logging.getLogger(__name__)


class RabbitVideoOffloaderV2(RabbitVideoOffloader):
    """
    Enhanced RabbitVideo offloader with v2 optimizations.

    Key differences from v1:
    - Uses AdaptiveBlockManagerV2 instead of BlockManager
    - Implements async prefetching
    - Learns and pins hot blocks
    - Batches transfers for efficiency
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_encoder_2: Optional[nn.Module] = None,
        max_gpu_blocks: int = 5,
        initial_gpu_blocks: int = 5,
        gpu_device: torch.device = torch.device("cuda:0"),
        enable_kv_cache: bool = False,
        enable_prefetch: bool = True,
        enable_smart_pinning: bool = True,
        enable_batching: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize RabbitVideo v2 offloader.

        Args:
            transformer: Main transformer model
            vae: VAE model (optional)
            text_encoder: Text encoder (optional)
            text_encoder_2: Second text encoder (optional)
            max_gpu_blocks: Max blocks on GPU during inference
            initial_gpu_blocks: Blocks to keep initially
            gpu_device: Target GPU device
            enable_kv_cache: Enable KV cache (v1.1 feature)
            enable_prefetch: Enable async prefetching (v2.0)
            enable_smart_pinning: Enable smart block pinning (v2.0)
            enable_batching: Enable batched transfers (v2.0)
            verbose: Enable verbose logging
        """
        # Don't call super().__init__ - we'll initialize manually
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.gpu_device = gpu_device
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Initialize core components
        self.memory_monitor = MemoryMonitor(gpu_device)
        self.tracker = BlockTracker()

        # Use v2 block manager
        self.block_manager = AdaptiveBlockManagerV2(
            tracker=self.tracker,
            memory_monitor=self.memory_monitor,
            max_gpu_blocks=max_gpu_blocks,
            gpu_device=gpu_device,
            enable_prefetch=enable_prefetch,
            enable_smart_pinning=enable_smart_pinning,
            enable_batching=enable_batching,
        )

        # Track auxiliary models
        self.vae_on_gpu = True
        self.text_encoder_on_gpu = True
        self.text_encoder_2_on_gpu = True

        logger.info("=" * 60)
        logger.info("RabbitVideo v2.0 Memory Optimization System")
        logger.info("=" * 60)
        logger.info("V2 Features:")
        logger.info(f"  - Async Prefetching: {enable_prefetch}")
        logger.info(f"  - Smart Block Pinning: {enable_smart_pinning}")
        logger.info(f"  - Batched Transfers: {enable_batching}")
        logger.info(f"  - KV Cache: {enable_kv_cache}")
        logger.info("=" * 60)

        # Phase 1: Smart initialization
        self._phase1_smart_initialization(initial_gpu_blocks)

        # Phase 2: Install hooks with v2 manager
        self._install_hooks_v2()

        logger.info("RabbitVideo v2.0 initialization complete")
        self.memory_monitor.log_stats("Post-init ")

    def _install_hooks_v2(self):
        """Install forward hooks with v2 support (prefetching, pinning, etc)."""
        logger.info("Phase 2: Installing v2 dynamic block loading hooks")

        # Track current denoising step for smart pinning
        self.current_step = 0

        for block_idx in self.tracker.blocks:
            block_info = self.tracker.blocks[block_idx]

            def make_pre_hook(idx):
                def pre_hook(module, input):
                    # V2: Use adaptive block manager
                    self.block_manager.ensure_block_on_gpu(idx)
                return pre_hook

            block_info.module.register_forward_pre_hook(make_pre_hook(block_idx))

        logger.info(f"Installed v2 hooks for {len(self.tracker.blocks)} blocks")

    def advance_denoising_step(self):
        """
        Call this at the start of each denoising step.

        This helps v2 features (smart pinning) learn access patterns.
        """
        self.current_step += 1
        self.block_manager.advance_step()

    def get_stats(self):
        """Get comprehensive statistics including v2 features."""
        base_stats = {
            "version": "2.0",
            "tracker": self.tracker.get_stats(),
            "memory": {
                "peak_gb": self.memory_monitor.peak_memory,
                "current": self.memory_monitor.get_stats().__dict__,
            },
            "auxiliary_models": {
                "vae_on_gpu": self.vae_on_gpu,
                "text_encoder_on_gpu": self.text_encoder_on_gpu,
                "text_encoder_2_on_gpu": self.text_encoder_2_on_gpu,
            }
        }

        # Add v2-specific stats
        v2_stats = self.block_manager.get_v2_stats()
        base_stats.update(v2_stats)

        return base_stats

    def print_stats(self):
        """Print comprehensive statistics including v2 features."""
        stats = self.get_stats()

        logger.info("=" * 60)
        logger.info("RabbitVideo v2.0 Statistics")
        logger.info("=" * 60)
        logger.info(f"Transformer Blocks: {stats['tracker']['total_blocks']}")
        logger.info(f"  On GPU: {stats['tracker']['gpu_blocks']}")
        logger.info(f"  On CPU: {stats['tracker']['cpu_blocks']}")
        logger.info(f"  Total Transfers: {stats['tracker']['total_transfers']}")
        logger.info(f"Peak Memory: {stats['memory']['peak_gb']:.2f} GB")

        # V2 stats
        if 'prefetch' in stats:
            pf_stats = stats['prefetch']
            logger.info(f"\nPrefetching:")
            logger.info(f"  Hit Rate: {pf_stats['hit_rate']:.1f}%")
            logger.info(f"  Hits: {pf_stats['prefetch_hits']}")
            logger.info(f"  Misses: {pf_stats['prefetch_misses']}")
            logger.info(f"  Total Time Saved: {pf_stats['total_prefetch_time']:.2f}s")

        if 'smart_pinning' in stats:
            pin_stats = stats['smart_pinning']
            logger.info(f"\nSmart Pinning:")
            logger.info(f"  Pinned Blocks: {pin_stats['pinned_blocks']}")
            logger.info(f"  Pinned IDs: {pin_stats['pinned_block_ids']}")
            logger.info(f"  Learning Complete: {pin_stats['learning_complete']}")

        logger.info("=" * 60)
