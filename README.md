# MLC-LLM from source on Jetson Orin Nano

These steps are a distillation of my journey to a proven, working build and deployment. It took me a while and several reinstalls to find the right flow for a smooth experience. This guide will provide you with everything to help you achieve:
- Building TVM from source on your Jetson
- Building MLC-LLM from source on your Jetson
- Converting a model from Hugging Face to MLC format and running it from command line and/or using a Python script

**Assumptions:**
- Jetpack 6.2 is installed
- An NVME is installed, formatted to EXT4 and partitioned. Make sure it's compatible with the Jetson and preferrably low power.

# 0. SETUP

## 0.1. Mount NVME drive

- Note: *In this step you decide what your NVME drive is mounted as. I use `/mnt/nvme`, some people prefer `/mnt/ssd`. Throughout this guide, `/mnt/nvme` is used. It's perfectly safe to replace this with your own name, but do this consistently for every instance*
- Note2: *Replace `[user]` in the chown command with your username*

``` bash
sudo mkdir /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
sudo chown -R [user]:[user] /mnt/nvme
sudo chmod -R 755 /mnt/nvme
echo '/dev/nvme0n1p1 /mnt/nvme ext4 defaults 0 2' | sudo tee -a /etc/fstab
```

## 0.2. .bashrc

During the installation steps, several modules and dependencies are installed manually, or via APT. In the end, your `.bashrc` should have these lines added at the end of the file. We can add them as a first step to make sure you always have the paths and variables available to make the installation smoother :-)

``` bash
export TVM_HOME=/mnt/nvme/tvm
export PYTHONPATH=/mnt/nvme/tvm/python:$PYTHONPATH
export PATH=/usr/local/cuda/bin:/usr/lib/llvm-17/bin:/home/eric/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/llvm-17/lib:/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

Activate your environment: `source ~/.bashrc`


# 1. SWAP SPACE CONFIG

- Allocate 32G of swap on it
- Update fstab to make permanent
- Disable swapfiles on microSD card

## 1.1. Create swap file on NVME

``` bash
sudo fallocate -l 32G /mnt/nvme/swapfile  # Create an 32GB file on NVMe
sudo chmod 600 /mnt/nvme/swapfile  # 600 ensures only root can read/write the swap file for security.
sudo mkswap /mnt/nvme/swapfile  # Format as swap
sudo swapon /mnt/nvme/swapfile  # Enable swap
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 1.2. Disable ZRAM

The Jetson is configured to have some swap files on the microSD card. We're using the NVME for that because it's a lot faster. This step is optional, but recommended to reduce read/write operations on your microSD card.

``` bash
sudo systemctl disable nvzramconfig
```

Verify with the following command (the zram swap files will stil show up until after next reboot):

``` bash
swapon --show
free -h
```

You can run `sudo reboot now` if you want to fully deactivate the zram, but it's not required.

# 2. Install general modules

First, we'll install everything we need in order to proceed. In this guide, I don't use a Python virtual environment to make things less confusing for new users. We'll install the dependency anyway if you prefer to use it.

``` bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip libopenblas-dev libtinfo-dev
```

# 3. TVM BUILD

A compatible TVM build **requires** LLVM-17, which isn't available via apt-get for the Jetson. We'll start by downloading and installing the binary before building TVM.

## 3.1. INSTALL LLVM-17

In this step, you will get the binary, unpack it and move it to `/usr/lib/llvm-17`. It's a large file, so it might take a while depending on your network connection.

``` bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-aarch64-linux-gnu.tar.xz
tar xf clang+llvm-17.0.6-aarch64-linux-gnu.tar.xz
sudo mv clang+llvm-17.0.6-aarch64-linux-gnu /usr/lib/llvm-17
```

## 3.2. BUILD TVM

These steps, you will pull the TVM source, configure the build and build your own TVM. We'll start off with cloning the source from GitHub:

``` bash
cd /mnt/nvme/
git clone --recursive https://github.com/apache/tvm.git && cd tvm
```

Clean up the build folder and copy a fresh config.cmake to it (If you ever want/need to make a fresh build, this place is where you can start off):

``` bash
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .
```

*At the time of writing, the commit hash used was `bfb0dd6a161d33c58f67469988244b139366e063`. If issues arise during step 3.2.2, try to use this hash and try again. If compilation still fails, there might be something wrong in your environment; retrace your steps and verify everything is installed correctly*

### 3.2.1. CONFIGURE BUILD

Inside the `/mnt/nvme/tvm/build` directory, we have a `config.cmake` file that is used to set flags; use the following to setup the config to use CUDA, Setup the correct CUDA architecture and use recommended settings for MLC-LLM. You can copy/paste this entire block and run it. The lines that start with '-' will be printed to your console and can be ignored; I added those as a means of commenting what everything is for.

**Note for Jetson Users**:
- Jetson Orin Nano uses **CUDA Architecture 87** (`Ampere`).
- Jetson AGX Orin uses **87 + 89** (if you have both GPU clusters).
- Verify your architecture with:
  
```bash
nvcc --version
```

*Note: We enable CUTLASS here for TVM, but MLC-LLM will disable it during its build (see Section 4.3). I'm still looking for a way to enable CUTLASS with MLC-LLM, and enabling it now will save us some time in the future.*

``` bash
echo - LLVM is a must dependency
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo - GPU SDKs, turn on if needed
echo "set(USE_CUDA   ON)" >> config.cmake
echo "set(USE_ROCM   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake
echo - Below are options for CUDA, turn on if needed
echo - CUDA_ARCH is the cuda compute capability of your GPU.
echo - Examples: 89 for 4090, 90a for H100/H200, 100a for B200.
echo - Reference: https://developer.nvidia.com/cuda-gpus
echo "set(CMAKE_CUDA_ARCHITECTURES 87)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake
echo "set(USE_THRUST ON)" >> config.cmake
echo "set(USE_NVTX OFF)" >> config.cmake
echo "set(USE_HIPBLAS OFF)" >> config.cmake
echo - Below is the option for ROCM, turn on if needed
echo "set(USE_RPC ON)" >> config.cmake
echo "set(USE_GRAPH_EXECUTOR ON)" >> config.cmake
```

### 3.2.2. COMPILE AND INSTALL

Now that we have the source code and the build configuration ready, it's time to build. This takes a while (15-20 minutes) on the Jetson. The following command is ran inside `/mnt/nvme/tvm/build`, you're probably still in this folder:

``` bash
cmake -DLLVM_CONFIG=/usr/lib/llvm-17/bin/llvm-config .. && make -j$(nproc) && cd ..
```

TVM relies on tvm-ffi, so we need to install that aswell:

``` bash
pip3 install setuptools_scm
pip3 install ninja
pip3 install cython
pip3 install psutil
cd /mnt/nvme/tvm/3rdparty/tvm-ffi
pip3 install --user --no-build-isolation .
```

### 3.2.3. VERIFY BUILD (NEED TO TEST THIS AS IT TOOK A LOT OF TRIES)

Check if we have all modules, the following command should show `libtvm_allvisible.so`, `libtvm_runtime.so` and `libtvm.so`:

``` bash
cd /mnt/nvme/tvm/build && ls -lh libtvm*
```

#### 3.2.3.1. Confirm that TVM is properly installed as a python package and provide the location of the TVM python package

``` bash
python3 -c "import tvm; print(tvm.__file__)"
```
*should print something like `/some-path/lib/python3.13/site-packages/tvm/__init__.py`*


#### 3.2.3.2. Verify build flags

``` bash
python3 -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```
*should print the build flags we've set in 3.2.1*


#### 3.2.3.3. Verify CUDA

``` bash
python3 -c "import tvm; print(tvm.cuda().exist)"
```
*should print `True`*


# 4. MLC-LLM BUILD

With TVM built and verified, we can move to MLC-LLM! 

## 4.1. Install RUST

Rust will be installed in this step, it will add itself to your `~/.bashrc` aswell (`. "$HOME/.cargo/env"` will be added) and enable itself with the source command. 

``` bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc
rustup default stable
```

## 4.2. Clone MLC-LLM

Clone the MLC-LLM source code:

``` bash
cd /mnt/nvme/
git clone --recursive https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
mkdir -p build && cd build
```

## 4.3. Configure build settings for MLC-LLM

In this step, we're creating a build configuration file. Make sure you're inside `/mnt/nvme/mlc-llm/build`.

When running gen_cmake_config, a couple of questions are asked. Make sure you reply with the following answers:
- TVM_SOURCE_DIR: Your TVM home directory (`/mnt/nvme/tvm` in this guide)
- Use CUDA: `y`

The rest of the answers should be answered with `n`. **We've built TVM with CUTLASS enabled and we're not using it here**. CUTLASS (CUDA Templates for Linear Algebra Subroutines) in a nutshell speeds up linear algebra operations which are basically the heart of transformer models. On the Jetson Orin Nano with its limited VRAM, I was unable to build MLC-LLM with CUTLASS enabled and chose to disable it. If, in the future I manage to build MLC-LLM with CUTLASS enabled, I will update this guide.


``` bash
python3 ../cmake/gen_cmake_config.py

Enter TVM_SOURCE_DIR in absolute path. If not specified, 3rdparty/tvm will be used by default: /mnt/nvme/tvm
Use CUDA? (y/n): y
Use CUTLASS? (y/n): n
Use CUBLAS? (y/n): n
Use ROCm? (y/n): n
Use Vulkan? (y/n): n
Use Metal (Apple M1/M2 GPU) ? (y/n): n
Use OpenCL? (y/n) n
```

## 4.4. Build MLC-LLM

In this step, we're going to build MLC-LLM. There's 1 thing we need to do though, and that is to relax the flashinfer version as it's set to use **exactly** 0.4.0 right now. This version isn't available via pip3, so we'll use another one by doing the following:

### 4.4.1. Update flashinfer dependency version

``` bash
cd /mnt/nvme/mlc-llm/python
```

Edit requirements.txt and change `flashinfer-python==0.4.0` to `flashinfer-python>=0.4.0`

*Mandatory disclaimer: Using >=0.4.0 may introduce compatibility issues. If you encounter errors, try flashinfer-python==0.4.0 with a manual wheel install. At the time of writing though, it worked perfectly!*

### 4.4.2. Install flashinfer

``` bash
pip3 install flashinfer-python
pip3 install -e .
```

### 4.4.3. Build MLC-LLM

``` bash
cd /mnt/nvme/mlc-llm/build/
cmake .. && make -j $(nproc) && cd ..
```

### 4.4.4. Verify build

Verify whether MLC-LLM is installed correctly by running:

``` bash
python3 -c "import mlc_llm; print(mlc_llm.__version__)"
```
*Should print something like `0.1.dev0`*

``` bash
python3 -c "from mlc_llm import cuda; print(cuda().exist)"
```
*Should print `True`*

Once you've reached this place, you're all set!


# 5. Troubleshooting

# 5. TROUBLESHOOTING
### Common Issues and Fixes
| Issue                          | Likely Cause                     | Solution                                  |
|--------------------------------|----------------------------------|-------------------------------------------|
| `cmake` fails with CUDA errors | Incorrect `CMAKE_CUDA_ARCHITECTURES` | Set to `87` for Jetson Orin Nano.         |
| `pip3 install` fails           | Missing dependencies             | Run `sudo apt-get install -y libopenmpi-dev libssl-dev`. |
| `import tvm` fails             | Python paths not set             | Verify `PYTHONPATH` in `.bashrc`.         |
| Out of memory during build     | Insufficient swap                | Verify `free -h` shows 32G swap.          |
| CUTLASS errors in MLC-LLM      | CUTLASS enabled in MLC-LLM       | Re-run `gen_cmake_config.py` with `n`.    |

