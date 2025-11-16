"""
Core RabbitVideo classes for memory management and block offloading.

This module implements the three-phase architecture:
- Phase 1: Smart Initialization (proactive offloading)
- Phase 2: Dynamic Block Swapping (LRU-based on-demand loading)
- Phase 3: Auxiliary Model Management (VAE and text encoder offloading)
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated: float  # GB
    reserved: float  # GB
    free: float  # GB
    total: float  # GB


class MemoryMonitor:
    """
    Monitor GPU memory usage and provide real-time statistics.

    This class tracks memory consumption throughout the inference process
    and helps identify optimization opportunities.
    """

    def __init__(self, device: torch.device):
        """
        Initialize memory monitor.

        Args:
            device: CUDA device to monitor
        """
        self.device = device
        self.device_index = device.index if device.index is not None else 0
        self.peak_memory = 0.0
        self.enabled = torch.cuda.is_available()

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not self.enabled:
            return MemoryStats(0, 0, 0, 0)

        allocated = torch.cuda.memory_allocated(self.device_index) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_index) / (1024**3)
        total = torch.cuda.get_device_properties(self.device_index).total_memory / (1024**3)
        free = total - reserved

        self.peak_memory = max(self.peak_memory, allocated)

        return MemoryStats(
            allocated=allocated,
            reserved=reserved,
            free=free,
            total=total
        )

    def log_stats(self, prefix: str = ""):
        """Log current memory statistics."""
        stats = self.get_stats()
        logger.info(
            f"{prefix}Memory: {stats.allocated:.2f}GB allocated, "
            f"{stats.reserved:.2f}GB reserved, {stats.free:.2f}GB free, "
            f"Peak: {self.peak_memory:.2f}GB"
        )

    def reset_peak(self):
        """Reset peak memory counter."""
        self.peak_memory = 0.0
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device_index)


@dataclass
class BlockInfo:
    """Information about a transformer block."""
    index: int
    module: nn.Module
    on_gpu: bool
    last_used: float
    transfer_count: int = 0


class BlockTracker:
    """
    Track transformer blocks and their current device locations.

    Maintains metadata about each block including:
    - Current location (GPU/CPU)
    - Last access time (for LRU)
    - Transfer count (for statistics)
    """

    def __init__(self):
        """Initialize block tracker."""
        self.blocks: Dict[int, BlockInfo] = {}
        self.gpu_blocks: Set[int] = set()
        self.cpu_blocks: Set[int] = set()

    def register_block(self, index: int, module: nn.Module, on_gpu: bool = False):
        """
        Register a transformer block.

        Args:
            index: Block index
            module: Block module
            on_gpu: Whether block is initially on GPU
        """
        self.blocks[index] = BlockInfo(
            index=index,
            module=module,
            on_gpu=on_gpu,
            last_used=time.time()
        )

        if on_gpu:
            self.gpu_blocks.add(index)
        else:
            self.cpu_blocks.add(index)

    def mark_gpu(self, index: int):
        """Mark block as being on GPU."""
        if index in self.blocks:
            self.blocks[index].on_gpu = True
            self.blocks[index].last_used = time.time()
            self.gpu_blocks.add(index)
            self.cpu_blocks.discard(index)

    def mark_cpu(self, index: int):
        """Mark block as being on CPU."""
        if index in self.blocks:
            self.blocks[index].on_gpu = False
            self.cpu_blocks.add(index)
            self.gpu_blocks.discard(index)

    def touch(self, index: int):
        """Update last access time for block."""
        if index in self.blocks:
            self.blocks[index].last_used = time.time()

    def increment_transfer(self, index: int):
        """Increment transfer count for block."""
        if index in self.blocks:
            self.blocks[index].transfer_count += 1

    def get_lru_gpu_block(self) -> Optional[int]:
        """Get least recently used block currently on GPU."""
        gpu_blocks_info = [
            self.blocks[idx] for idx in self.gpu_blocks
        ]

        if not gpu_blocks_info:
            return None

        lru_block = min(gpu_blocks_info, key=lambda b: b.last_used)
        return lru_block.index

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        total_transfers = sum(b.transfer_count for b in self.blocks.values())
        return {
            "total_blocks": len(self.blocks),
            "gpu_blocks": len(self.gpu_blocks),
            "cpu_blocks": len(self.cpu_blocks),
            "total_transfers": total_transfers,
        }


class BlockManager:
    """
    Manage dynamic block transfers between GPU and CPU with LRU eviction.

    This implements Phase 2 of RabbitVideo: on-demand block loading with
    intelligent eviction policy to minimize transfers.
    """

    def __init__(
        self,
        tracker: BlockTracker,
        memory_monitor: MemoryMonitor,
        max_gpu_blocks: int = 5,
        gpu_device: torch.device = torch.device("cuda:0"),
        enable_kv_cache: bool = False,
    ):
        """
        Initialize block manager.

        Args:
            tracker: Block tracker instance
            memory_monitor: Memory monitor instance
            max_gpu_blocks: Maximum number of blocks to keep on GPU
            gpu_device: Target GPU device
            enable_kv_cache: Enable experimental KV cache optimization
        """
        self.tracker = tracker
        self.memory_monitor = memory_monitor
        self.max_gpu_blocks = max_gpu_blocks
        self.gpu_device = gpu_device
        self.enable_kv_cache = enable_kv_cache
        self.kv_cache: Dict[int, Tuple] = {} if enable_kv_cache else None

        logger.info(f"BlockManager initialized: max_gpu_blocks={max_gpu_blocks}, kv_cache={enable_kv_cache}")

    def _synchronous_transfer_to_gpu(self, block: nn.Module) -> None:
        """
        Transfer block to GPU with synchronous operations.

        Critical: PyTorch's async operations don't immediately free memory.
        We use synchronize() and empty_cache() to ensure memory is actually freed.
        """
        block.to(self.gpu_device)
        torch.cuda.synchronize()

    def _synchronous_transfer_to_cpu(self, block: nn.Module) -> None:
        """
        Transfer block to CPU with synchronous operations and aggressive cache clearing.

        Double clearing ensures PyTorch actually releases GPU memory.
        """
        block.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Double clearing for safety

    def ensure_block_on_gpu(self, block_index: int) -> None:
        """
        Ensure specified block is on GPU, using LRU eviction if needed.

        Args:
            block_index: Index of block to load
        """
        # Already on GPU?
        if block_index in self.tracker.gpu_blocks:
            self.tracker.touch(block_index)
            return

        # Need to evict a block?
        while len(self.tracker.gpu_blocks) >= self.max_gpu_blocks:
            lru_index = self.tracker.get_lru_gpu_block()
            if lru_index is None:
                break

            # Don't evict the block we're about to load
            if lru_index == block_index:
                break

            self._offload_block(lru_index)

        # Load the block
        self._load_block(block_index)

    def _load_block(self, block_index: int) -> None:
        """Load block from CPU to GPU."""
        if block_index not in self.tracker.blocks:
            logger.warning(f"Block {block_index} not registered")
            return

        block_info = self.tracker.blocks[block_index]

        logger.debug(f"Loading block {block_index} to GPU")
        start_time = time.time()

        self._synchronous_transfer_to_gpu(block_info.module)

        self.tracker.mark_gpu(block_index)
        self.tracker.increment_transfer(block_index)

        transfer_time = time.time() - start_time
        logger.debug(f"Block {block_index} loaded in {transfer_time:.3f}s")

    def _offload_block(self, block_index: int) -> None:
        """Offload block from GPU to CPU."""
        if block_index not in self.tracker.blocks:
            logger.warning(f"Block {block_index} not registered")
            return

        block_info = self.tracker.blocks[block_index]

        logger.debug(f"Offloading block {block_index} to CPU")
        start_time = time.time()

        self._synchronous_transfer_to_cpu(block_info.module)

        self.tracker.mark_cpu(block_index)

        transfer_time = time.time() - start_time
        logger.debug(f"Block {block_index} offloaded in {transfer_time:.3f}s")

    def offload_all_blocks(self) -> None:
        """Offload all blocks to CPU."""
        logger.info("Offloading all blocks to CPU")
        gpu_blocks = list(self.tracker.gpu_blocks)
        for block_index in gpu_blocks:
            self._offload_block(block_index)

    def get_cache_stats(self) -> Dict:
        """Get KV cache statistics."""
        if not self.enable_kv_cache or self.kv_cache is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "cached_blocks": len(self.kv_cache),
            "cache_size_mb": sum(
                sum(t.element_size() * t.nelement() for t in tensors) / (1024**2)
                for tensors in self.kv_cache.values()
            )
        }


class RabbitVideoOffloader:
    """
    Main orchestrator for RabbitVideo memory optimization.

    Implements the three-phase architecture:
    1. Smart Initialization: Proactively offload most blocks at startup
    2. Dynamic Block Swapping: Load blocks on-demand with LRU eviction
    3. Auxiliary Model Management: Offload VAE and text encoders during inference
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
        verbose: bool = True,
    ):
        """
        Initialize RabbitVideo offloader.

        Args:
            transformer: Main transformer model with blocks to offload
            vae: VAE model (optional, for Phase 3)
            text_encoder: Text encoder model (optional, for Phase 3)
            text_encoder_2: Second text encoder model (optional, for Phase 3)
            max_gpu_blocks: Maximum blocks to keep on GPU during inference
            initial_gpu_blocks: Number of blocks to keep on GPU initially
            gpu_device: Target GPU device
            enable_kv_cache: Enable experimental KV cache optimization (v1.1)
            verbose: Enable verbose logging
        """
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.gpu_device = gpu_device
        self.verbose = verbose

        # Set logging level
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Initialize core components
        self.memory_monitor = MemoryMonitor(gpu_device)
        self.tracker = BlockTracker()
        self.block_manager = BlockManager(
            tracker=self.tracker,
            memory_monitor=self.memory_monitor,
            max_gpu_blocks=max_gpu_blocks,
            gpu_device=gpu_device,
            enable_kv_cache=enable_kv_cache,
        )

        # Phase 3: Track auxiliary model states
        self.vae_on_gpu = True
        self.text_encoder_on_gpu = True
        self.text_encoder_2_on_gpu = True

        logger.info("=" * 60)
        logger.info("RabbitVideo Memory Optimization System")
        logger.info(f"Version: 1.1.0 {'(with KV cache)' if enable_kv_cache else ''}")
        logger.info("=" * 60)

        # Phase 1: Smart Initialization
        self._phase1_smart_initialization(initial_gpu_blocks)

        # Install forward hooks for Phase 2
        self._install_hooks()

        logger.info("RabbitVideo initialization complete")
        self.memory_monitor.log_stats("Post-init ")

    def _phase1_smart_initialization(self, initial_gpu_blocks: int):
        """
        Phase 1: Proactively offload most transformer blocks to CPU.

        Args:
            initial_gpu_blocks: Number of blocks to keep on GPU initially
        """
        logger.info("Phase 1: Smart Initialization")
        self.memory_monitor.log_stats("Pre-offload ")

        # Find transformer blocks
        blocks = self._find_transformer_blocks()

        if not blocks:
            logger.warning("No transformer blocks found! RabbitVideo may not work correctly.")
            return

        logger.info(f"Found {len(blocks)} transformer blocks")

        # Register all blocks
        for idx, block in enumerate(blocks):
            self.tracker.register_block(idx, block, on_gpu=True)

        # Offload blocks beyond the initial GPU blocks
        blocks_to_offload = len(blocks) - initial_gpu_blocks
        logger.info(f"Offloading {blocks_to_offload}/{len(blocks)} blocks to CPU")

        for idx in range(initial_gpu_blocks, len(blocks)):
            block = self.tracker.blocks[idx].module
            self.block_manager._synchronous_transfer_to_cpu(block)
            self.tracker.mark_cpu(idx)

        self.memory_monitor.log_stats("Post-offload ")
        logger.info(f"Phase 1 complete: {len(self.tracker.gpu_blocks)} blocks on GPU, "
                   f"{len(self.tracker.cpu_blocks)} blocks on CPU")

    def _find_transformer_blocks(self) -> List[nn.Module]:
        """
        Find transformer blocks in the model.

        Searches for common block patterns:
        - transformer_blocks (CogVideoX)
        - blocks (generic)
        - layers (some architectures)
        """
        blocks = []

        # Try common attribute names
        for attr_name in ['transformer_blocks', 'blocks', 'layers']:
            if hasattr(self.transformer, attr_name):
                blocks_attr = getattr(self.transformer, attr_name)
                if isinstance(blocks_attr, nn.ModuleList) or isinstance(blocks_attr, list):
                    blocks = list(blocks_attr)
                    logger.info(f"Found blocks via '{attr_name}' attribute")
                    break

        # If no blocks found, try to find them recursively
        if not blocks:
            for name, module in self.transformer.named_modules():
                if 'block' in name.lower() and isinstance(module, nn.Module):
                    # Avoid adding container modules
                    if not list(module.children()):
                        continue
                    blocks.append(module)

        return blocks

    def _install_hooks(self):
        """Install forward hooks for dynamic block loading (Phase 2)."""
        logger.info("Phase 2: Installing dynamic block loading hooks")

        for block_idx in self.tracker.blocks:
            block_info = self.tracker.blocks[block_idx]

            def make_pre_hook(idx):
                def pre_hook(module, input):
                    # Ensure this block is on GPU before forward pass
                    self.block_manager.ensure_block_on_gpu(idx)
                return pre_hook

            block_info.module.register_forward_pre_hook(make_pre_hook(block_idx))

        logger.info(f"Installed hooks for {len(self.tracker.blocks)} blocks")

    def offload_text_encoders(self):
        """Phase 3: Offload text encoders to CPU to save memory during inference."""
        logger.info("Phase 3: Offloading text encoders to CPU")

        if self.text_encoder is not None and self.text_encoder_on_gpu:
            self.text_encoder.to('cpu')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.text_encoder_on_gpu = False
            logger.info("Text encoder offloaded")

        if self.text_encoder_2 is not None and self.text_encoder_2_on_gpu:
            self.text_encoder_2.to('cpu')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.text_encoder_2_on_gpu = False
            logger.info("Text encoder 2 offloaded")

        self.memory_monitor.log_stats("After text encoder offload ")

    def offload_vae(self):
        """Phase 3: Offload VAE to CPU during transformer inference."""
        if self.vae is not None and self.vae_on_gpu:
            logger.info("Phase 3: Offloading VAE to CPU")
            self.vae.to('cpu')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.vae_on_gpu = False
            self.memory_monitor.log_stats("After VAE offload ")

    def load_vae(self):
        """Phase 3: Load VAE back to GPU for decoding."""
        if self.vae is not None and not self.vae_on_gpu:
            logger.info("Phase 3: Loading VAE to GPU")

            # Offload all transformer blocks first to make room
            self.block_manager.offload_all_blocks()

            self.vae.to(self.gpu_device)
            torch.cuda.synchronize()
            self.vae_on_gpu = True
            self.memory_monitor.log_stats("After VAE load ")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            "tracker": self.tracker.get_stats(),
            "memory": {
                "peak_gb": self.memory_monitor.peak_memory,
                "current": self.memory_monitor.get_stats().__dict__,
            },
            "kv_cache": self.block_manager.get_cache_stats(),
            "auxiliary_models": {
                "vae_on_gpu": self.vae_on_gpu,
                "text_encoder_on_gpu": self.text_encoder_on_gpu,
                "text_encoder_2_on_gpu": self.text_encoder_2_on_gpu,
            }
        }

    def print_stats(self):
        """Print comprehensive statistics."""
        stats = self.get_stats()

        logger.info("=" * 60)
        logger.info("RabbitVideo Statistics")
        logger.info("=" * 60)
        logger.info(f"Transformer Blocks: {stats['tracker']['total_blocks']}")
        logger.info(f"  On GPU: {stats['tracker']['gpu_blocks']}")
        logger.info(f"  On CPU: {stats['tracker']['cpu_blocks']}")
        logger.info(f"  Total Transfers: {stats['tracker']['total_transfers']}")
        logger.info(f"Peak Memory: {stats['memory']['peak_gb']:.2f} GB")

        if stats['kv_cache']['enabled']:
            logger.info(f"KV Cache: {stats['kv_cache']['cached_blocks']} blocks, "
                       f"{stats['kv_cache']['cache_size_mb']:.1f} MB")

        logger.info("=" * 60)
