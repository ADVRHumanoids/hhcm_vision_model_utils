# Utility Tools

This directory contains general-purpose utility scripts for PyTorch model development and GPU memory management.

## Tools

### clearCuda.py

Clear CUDA cache and display detailed GPU memory statistics.

**Purpose**: Free unused GPU memory and inspect memory usage for debugging out-of-memory (OOM) errors.

**How It Works**:
1. Calls `torch.cuda.empty_cache()` to release unused cached memory
2. Prints comprehensive memory summary including allocations, cache, and fragmentation

**When to Use**:
- After training/evaluation to free memory before next experiment
- Debugging CUDA out-of-memory errors
- Monitoring GPU memory usage during development
- Before running memory-intensive operations

**Usage**:
```bash
python3 clearCuda.py
```

**Requirements**:
```bash
pip install torch
```

**Notes**:
- Only clears **unused** cached memory (active allocations are preserved)
- Does not reduce memory of active tensors
- Requires CUDA-capable GPU
- If no GPU available, prints empty summary

---

### modelInfo.py

Load and display PyTorch model architecture and device information.

**Purpose**: Quickly inspect saved PyTorch models to view architecture, layer details, and verify model structure.

**How It Works**:
1. Validates model file exists
2. Detects available device (CUDA or CPU)
3. Loads model from specified .pt file
4. Moves model to GPU if available
5. Prints complete model architecture

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model` | str | Yes | Path to the .pt model file |

**Usage**:
```bash
# Inspect a model file
python3 modelInfo.py --model path/to/model.pt
```

**Requirements**:
```bash
pip install torch
```

**Notes**:
- Model must be saved as .pt file using `torch.save()`
- Automatically handles CUDA availability
- Primarily for development/debugging (not production use)
- An alternative might be [torchinfo.summary](https://github.com/TylerYep/torchinfo)

---

## Dependencies

```bash
pip install torch
```

**CUDA Requirements**:
- NVIDIA GPU with CUDA support
- Matching CUDA Toolkit version
- PyTorch built with CUDA support

---

## Support and Resources

- **PyTorch CUDA Memory Management**: [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- **torch.cuda.empty_cache()**: [https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html)
- **Model Inspection**: [https://github.com/TylerYep/torchinfo](https://github.com/TylerYep/torchinfo)
- **Debugging OOM Errors**: [https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory](https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory)
