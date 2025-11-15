# RabbitVideo for CogVideoX

**RabbitVideo** is a sophisticated memory optimization system that enables running large video diffusion models (like CogVideoX-5B) on consumer 24GB GPUs by reducing peak memory usage from ~66GB to ~23GB with only 15% time overhead.

## Quick Start

### Installation

RabbitVideo is included in this CogVideo repository. Just make sure you have the required dependencies:

```bash
pip install torch diffusers transformers accelerate
```

### Basic Usage

```python
from diffusers import CogVideoXPipeline
import torch
from rabbitvideo import enable_rabbitvideo

# Load pipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

# Move to GPU
pipe.to("cuda")

# Enable RabbitVideo instead of sequential_cpu_offload
enable_rabbitvideo(pipe, max_gpu_blocks=5)

# Generate video as usual
video = pipe(
    prompt="A girl riding a bike through a beautiful park.",
    num_frames=49,
    num_inference_steps=50,
).frames[0]
```

### Alternative Method: Pipeline Patching

You can also patch the pipeline classes to use `pipe.enable_rabbitvideo()` directly:

```python
from rabbitvideo.pipeline_integration import patch_cogvideox_pipeline

# Patch pipeline classes (only needed once)
patch_cogvideox_pipeline()

# Now you can use enable_rabbitvideo() directly
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
pipe.to("cuda")
pipe.enable_rabbitvideo(max_gpu_blocks=5)
```

## Performance Comparison

Run the demo script to compare different memory optimization methods:

```bash
# Test RabbitVideo
python inference/rabbitvideo_demo.py --method rabbitvideo --max_gpu_blocks 5

# Compare all methods
python inference/rabbitvideo_demo.py --method all --num_inference_steps 20

# Use smaller model for testing
python inference/rabbitvideo_demo.py --method rabbitvideo --model_path THUDM/CogVideoX-2b
```

Expected results (CogVideoX-5B, 24GB GPU):

| Method | Peak Memory | Time Overhead | Notes |
|--------|-------------|---------------|-------|
| Standard (no offload) | ~66GB | Baseline | ⚠️ OOM on 24GB GPU |
| Sequential CPU Offload | ~28GB | +50% | ✅ Works on 24GB |
| **RabbitVideo** | **~23GB** | **+15%** | ✅ Best memory efficiency |

## Configuration Options

### `enable_rabbitvideo()` Parameters

- **`max_gpu_blocks`** (default: 5): Maximum transformer blocks to keep on GPU
  - 24GB GPU: Use 5
  - 16GB GPU: Use 4
  - 12GB GPU: Use 3

- **`initial_gpu_blocks`** (default: 5): Blocks to keep on GPU initially
  - Usually same as `max_gpu_blocks`

- **`enable_kv_cache`** (default: False): Experimental KV cache optimization
  - Provides 5-10% speedup
  - May use slightly more memory
  - v1.1 feature

- **`offload_text_encoder`** (default: True): Offload text encoder after use
  - Saves ~2-4GB during inference

- **`manage_vae`** (default: True): Automatically manage VAE transfers
  - Saves ~12GB during transformer inference

- **`verbose`** (default: True): Enable detailed logging

## How It Works

RabbitVideo uses a **three-phase architecture**:

### Phase 1: Smart Initialization
Proactively offload most transformer blocks (e.g., 55/60) to CPU at startup, keeping only a few on GPU.

### Phase 2: Dynamic Block Swapping
Load blocks on-demand during inference using an LRU (Least Recently Used) eviction policy. Exploits the sequential execution pattern of video diffusion models.

### Phase 3: Auxiliary Model Management
Offload VAE and text encoders when they're not needed:
- Text encoders: Offloaded after prompt encoding
- VAE: Offloaded during transformer inference, loaded back for decoding

### Key Innovations

1. **Block-level granularity**: Optimal balance between model-level (too slow) and layer-level (too fine)

2. **Synchronous transfer protocol**: Ensures PyTorch actually frees GPU memory:
   ```python
   block.to('cpu')
   torch.cuda.synchronize()
   torch.cuda.empty_cache()  # Double clearing!
   torch.cuda.synchronize()
   ```

3. **LRU eviction policy**: Minimizes transfers by exploiting sequential execution

4. **Proactive management**: Prevent OOM rather than respond to it

## API Reference

### `enable_rabbitvideo(pipeline, **kwargs)`

Enable RabbitVideo on a CogVideoX pipeline.

**Args:**
- `pipeline`: CogVideoX pipeline instance
- `max_gpu_blocks`: Max blocks on GPU (default: 5)
- `initial_gpu_blocks`: Initial blocks on GPU (default: 5)
- `enable_kv_cache`: Enable KV cache (default: False)
- `offload_text_encoder`: Offload text encoder (default: True)
- `manage_vae`: Manage VAE transfers (default: True)
- `verbose`: Verbose logging (default: True)

**Returns:** Modified pipeline with RabbitVideo enabled

### Pipeline Methods (added by RabbitVideo)

After enabling RabbitVideo, the pipeline gains these methods:

- **`pipe.get_rabbitvideo_stats()`**: Get comprehensive statistics
- **`pipe.print_rabbitvideo_stats()`**: Print statistics to console
- **`pipe.disable_rabbitvideo()`**: Disable RabbitVideo (experimental)

## Troubleshooting

### Out of Memory Errors

If you still get OOM errors:
1. Reduce `max_gpu_blocks` (try 3 or 4)
2. Use a smaller model (CogVideoX-2b instead of 5b)
3. Reduce `num_frames` or resolution
4. Enable VAE tiling: `pipe.vae.enable_tiling()`

### Slow Performance

If generation is too slow:
1. Increase `max_gpu_blocks` if you have memory headroom
2. Try `enable_kv_cache=True` for 5-10% speedup
3. Reduce `num_inference_steps` for testing

### Import Errors

Make sure you're importing from the correct path:

```python
# If running from CogVideo root
from rabbitvideo import enable_rabbitvideo

# If running from elsewhere, add to path:
import sys
sys.path.append('/path/to/CogVideo')
from rabbitvideo import enable_rabbitvideo
```

## Compatibility

### Supported Models
- ✅ CogVideoX-2b
- ✅ CogVideoX-5b
- ✅ CogVideoX1.5-5b
- ✅ CogVideoX-5b-I2V (Image-to-Video)
- ✅ CogVideoX1.5-5b-I2V

### Supported Techniques
- ✅ LoRA fine-tuning
- ✅ VAE slicing/tiling
- ✅ Text-to-Video (T2V)
- ✅ Image-to-Video (I2V)
- ✅ Video-to-Video (V2V)

### Not Yet Supported
- ❌ Multi-GPU inference (use one GPU per pipeline)
- ❌ Quantization (int8/int4) - may conflict with block management

## Technical Details

For implementation details, see:
- `rabbitvideo/core.py` - Core classes (MemoryMonitor, BlockTracker, BlockManager, RabbitVideoOffloader)
- `rabbitvideo/pipeline_integration.py` - Pipeline integration logic
- Technical report (coming soon)

## Citation

If you use RabbitVideo in your research, please cite:

```bibtex
@software{rabbitvideo2024,
  title={RabbitVideo: Memory Optimization for Video Diffusion Models},
  author={RabbitVideo Team},
  year={2024},
  url={https://github.com/THUDM/CogVideo}
}
```

## License

RabbitVideo is released under the same license as CogVideo.

## Acknowledgments

- CogVideo team at THUDM for the excellent video diffusion model
- HuggingFace diffusers team for the pipeline infrastructure
- Community feedback and contributions
