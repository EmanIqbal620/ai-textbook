---
sidebar_position: 1
---

# Prerequisites and Setup Instructions

Before starting this textbook, you'll need to set up your development environment with the necessary tools and frameworks.

## Software Prerequisites

- **Operating System**: Ubuntu 22.04 LTS or Windows 10/11 with WSL2
- **Basic Knowledge**:
  - Familiarity with Linux/Ubuntu environment
  - Basic Python programming skills
  - Understanding of linear algebra and calculus
  - Basic knowledge of control systems
  - Git version control experience

## Installation Guide

### 1. Install Ubuntu 22.04 LTS or set up WSL2 on Windows

If using Windows, install WSL2 with Ubuntu 22.04:

```bash
wsl --install Ubuntu-22.04
```

### 2. Install ROS 2 Humble Hawksbill

Follow the official installation guide for your OS:
- [ROS 2 Installation Guide](https://docs.ros.org/en/humble/Installation.html)

### 3. Set up NVIDIA drivers and CUDA toolkit

For Isaac Sim support, install the latest NVIDIA drivers and CUDA toolkit:
- Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### 4. Install Gazebo Garden and Unity Hub

- Gazebo Garden: Follow installation instructions at [Gazebo's website](https://gazebosim.org/docs/garden/install)
- Unity Hub: Download from [Unity's website](https://unity.com/)

### 5. Configure Isaac Sim environment

- Download Isaac Sim from NVIDIA Developer website
- Follow setup instructions in the Isaac Sim documentation

### 6. Install required Python packages and dependencies

```bash
pip3 install numpy scipy matplotlib
```

## Hardware Requirements

For optimal performance, your system should meet at least the minimum specifications outlined in the [Hardware Requirements section](./module-5-hardware/hardware-specifications.md).

## Verification

After completing the setup, verify your installation by running:

```bash
ros2 --version
python3 --version
nvidia-smi  # If NVIDIA GPU is available
```