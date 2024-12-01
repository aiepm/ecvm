# ECVM

Efficient computer vision models. Focus of this repository:

1. Small computer vision modules
2. Architecture optimization
3. Efficient training on consumer-grade gpus
4. Efficient inference on consumer-grade gpus, mobile/edge devices and CPUs.

## Installation

See [this](https://pytorch.org/cppdocs/installing.html)

**IMPORTANT**: download https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-xxx.zip version, otherwise the project will fail to build with `undefined reference to cv::imread(std::string const&, int)` because of conflict with opencv.

Also don't forget to add downloaded libtorch to LD_LIBRARY_PATH.

## Current state

Just started, this is not going to be usable for quite a while.
