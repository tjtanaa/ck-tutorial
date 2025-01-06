# ck-tutorial

# First tutorial is prepacking the tensors

## Launch a docker
```bash
#!/bin/bash
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/ck-tutorial?:/app/ck-tutorial \
   -e HF_HOME="/app/model" \
   rocm/vllm-dev:20241205-tuned \
   bash
```

## Setup
```bash
sudo apt remove composablekernel-dev
git clone https://github.com/ROCm/composable_kernel.git --branch update_cka8w8_uc
cd composable_kernel
mkdir build
cd build
make -j64
make -j install
```

## Install torch_ck
```bash
python setup.py develop
```
 