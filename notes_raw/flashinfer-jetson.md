## FlashInfer sm87 Support for Jetson Orin AGX

### Prerequisites

- Jetson Orin AGX (or other sm87 hardware)
- CUDA Toolkit installed
- Python 3.10+
- Virtual environment (recommended)

## Step 1: Create Build Environment

Create folder `flashinfer-build` and set up virtual environment:

```bash
mkdir flashinfer-build
cd flashinfer-build
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install Dependencies

Install build dependencies:

```bash
pip3 install --upgrade pip setuptools wheel
pip3 install cmake ninja
```

Install CUDA-enabled PyTorch:

**Important:** Use the CUDA version that matches your system. For CUDA 12.6, use `cu126`:

```bash
pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Note:** If your system has CUDA 13.0, use `cu130` instead. Check your CUDA version with `nvcc --version`.

## Step 3: Clone and Build FlashInfer

Clone FlashInfer repository:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
```

Set architecture for sm87 and build:

```bash
export FLASHINFER_CUDA_ARCH_LIST="8.7"
pip install -v . --no-build-isolation
```

## Step 4: Understanding Build vs Runtime

**Important:** Building FlashInfer from source with `FLASHINFER_CUDA_ARCH_LIST="8.7"` compiles kernels for sm87, but does NOT change the runtime architecture check.

When you run:
```bash
export FLASHINFER_CUDA_ARCH_LIST="8.7"
pip install -v . --no-build-isolation
```

**What it does:**
- ✅ Compiles CUDA kernels for sm87 architecture
- ✅ Creates FlashInfer package with sm87 kernels included
- ❌ **Does NOT modify the runtime check logic**

**What it does NOT do:**
- ❌ Does NOT "bake in" sm87 support into the runtime check
- ❌ Does NOT change `CompilationContext` behavior
- ❌ Does NOT modify `check_cuda_arch()` function

This means you still need to either:
1. Set `FLASHINFER_CUDA_ARCH_LIST="8.7"` every time you use FlashInfer, OR
2. Apply the patch (recommended)

## Step 5: Configure Runtime Environment

You have two options to ensure FlashInfer works at runtime:

### Set Environment Variable in `.bashrc`

Add the environment variable to your shell configuration so it's always available:

```bash
echo 'export FLASHINFER_CUDA_ARCH_LIST="8.7"' >> ~/.bashrc
source ~/.bashrc
```

**Pros:**
- Clean and simple solution
- Always available in new shell sessions
- No patching required
- Works system-wide

**Cons:**
- Requires shell restart or `source ~/.bashrc` for new sessions

### Option B: Apply the Patch

Alternatively, you can patch FlashInfer to check actual GPU capability as a fallback:

```bash
cd /home/eric/projects/misty/flashinfer-build
source .venv/bin/activate
python3 patch_flashinfer_sm87.py
```

**Note:** The patch may not work in all scenarios. If you encounter issues, use Option A instead.

**What the patch does:**
- Modifies `check_cuda_arch()` to check actual GPU capability as fallback
- Intended to work even if environment variable isn't set

**Pros:**
- Works automatically (when it works)
- No need to remember environment variable

**Cons:**
- Requires patching installed FlashInfer
- May not work in all scenarios

## Step 6: Verify Configuration

### If using Option A (.bashrc):

Verify the environment variable is set:

```bash
echo $FLASHINFER_CUDA_ARCH_LIST
# Should output: 8.7
```

### If using Option B (Patch):

Verify the patch works:

```bash
python3 -c "from flashinfer.jit.core import check_cuda_arch; import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute: {torch.cuda.get_device_capability(0)}'); check_cuda_arch(); print('✅ Patch verified - FlashInfer enabled')"
```

Expected output:
```
GPU: NVIDIA Jetson Orin AGX
Compute: (8, 7)
✅ Patch verified - FlashInfer enabled
```

## Step 7: Compile Model with FlashInfer

Now you can compile your MLC-LLM model with FlashInfer support:

```bash
mlc_llm compile <model-config.json> \
  --device cuda \
  --opt O3 \
  -o <output>.so
```

**What to look for in the logs:**

✅ **Success indicators:**
- No "FlashInfer requires GPUs with sm75 or higher" error
- "FlashInfer PagedKVCache created successfully" (or similar)
- Model compiles without falling back to TIR-based KV cache

❌ **If you see errors:**
- "FlashInfer requires GPUs with sm75 or higher" → Patch may not be applied correctly
- "not enough values to unpack" → FlashInfer version mismatch (use 0.5.3 or 0.6.3)

## Summary

This workflow gives you:
- ✅ sm87 kernels compiled (from build)
- ✅ Runtime check works via environment variable (from `.bashrc`) or patch
- ✅ ~20-30% performance improvement expected

**Recommended approach:** Use Option A (`.bashrc`) for a clean, reliable solution.

## Troubleshooting

### Build fails with "bdist_wheel" error
```bash
pip3 install --upgrade pip setuptools wheel
```

### Patch script not found
Make sure you're in the `flashinfer-build` directory and the patch script exists:
```bash
ls -la patch_flashinfer_sm87.py
```

### FlashInfer still fails after patch
**Solution:** Use Option A (add to `.bashrc`) instead - it's more reliable.

If you want to troubleshoot the patch:
1. Verify patch was applied:
   ```bash
   python3 -c "from flashinfer.jit.core import check_cuda_arch; check_cuda_arch()"
   ```

2. Check FlashInfer location:
   ```bash
   python3 -c "import flashinfer; print(flashinfer.__file__)"
   ```

3. Re-apply patch if needed:
   ```bash
   python3 patch_flashinfer_sm87.py
   ```

### Environment variable not set in new terminal
If you added it to `.bashrc`, make sure to:
- Restart your terminal, OR
- Run `source ~/.bashrc` in the current session

