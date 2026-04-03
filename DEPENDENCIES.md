# Project Dependencies
This project depends on the following system packages. Please ensure they are installed before building or running the project.

## Required Packages

| Package Name      | Description                          | Installation Command (Debian/Ubuntu) |
|-------------------|--------------------------------------|--------------------------------------|
| build-essential   | Common C++ packages                  | `sudo apt install build-essential`   |
| cgal              | The Computational Geometry Algorithms Library | `sudo apt install libcgal-dev`       |
| spdlog            | Fast C++ logging library             | `sudo apt install libspdlog-dev`     |
| cuda-toolkit      | Nvidia CUDA Toolkit (manual install) | [Download from NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu-installation) |
| optix             | Nvidia OptiX (manual install)        | [Download from NVIDIA](https://developer.nvidia.com/optix) |

## Installation Instructions

### Debian/Ubuntu

Run the following commands to install all dependencies except OptiX (which must be downloaded and installed manually):

```bash
sudo apt update
sudo apt install -y build-essential libcgal-dev libspdlog-dev
```

After installing the above packages, proceed to manually install the Nvidia dependencies (CUDA Toolkit and OptiX). Good luck! 🤞🤞