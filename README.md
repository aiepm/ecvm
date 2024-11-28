# ECVM

Efficient computer vision models. Focus of this repository:

    1. Small computer vision modules
    2. Architecture optimization
    3. Efficient training on consumer-grade gpus
    4. Efficient inference on consumer-grade gpus, mobile/edge devices and CPUs.

## Installation

If torch is installed with pip
```bash
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
```

Otherwise see [this](https://pytorch.org/cppdocs/installing.html)

## Current state

Just started, this is not going to be usable for quite a while.
