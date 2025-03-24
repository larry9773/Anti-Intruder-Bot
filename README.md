# Autonomous Target Navigation & Anti-Intruder Bot (AIB)

## Introduction

This project aims to develop an autonomous robot using Turtlebot 4 that is capable of:
- **Target Recognition**: Detecting and identifying intruders or objects of interest in real-time with a custom-trained neural network.
- **Autonomous Navigation**: Patrolling an assigned area efficiently by fusing sensor data (LiDAR, cameras, IMU, etc.) for optimal navigation.
- **Dynamic Mapping**: Generating and updating a real-time map with visualized covered areas (highlighted in green) to ensure complete area coverage and prompt intrusion detection.

## Features

- **Intelligent Detection**: Utilizes a neural network to process image data and perform target recognition.
- **Optimized Navigation**: Implements an autonomous navigation system based on ROS2 and Nav2 (or a custom algorithm) to maximize area coverage.
- **Reinforcement Learning**: Uses the Proximal Policy Optimization (PPO) algorithm to train the robot in simulated environments (Gazebo), improving decision-making.
- **Multi-Sensor Fusion**: Integrates data from LiDAR, RGB camera, stereo depth sensor, and IMU for robust environment perception.
- **Coverage Visualization**: Marks visited regions on the LiDAR-generated map to validate effective patrol and intrusion clearance.

## Project Architecture

The system is divided into several interconnected modules:
- **Image Recognition Module**: Processes and analyzes video feeds to identify potential targets.
- **Autonomous Navigation Module**: Handles path planning and robot control using sensor data.
- **Sensor Fusion Module**: Combines data from various sensors (LiDAR, cameras, etc.) to enhance environmental understanding.
- **Reinforcement Learning Module**: Trains the robot using the PPO algorithm with rewards based on area coverage and target detection performance.

## Installation

### Prerequisites

- **ROS2** (compatible with Turtlebot 4)
- **Nav2** (for basic navigation; optional if using a custom algorithm)
- **Gazebo** (for simulation and training)
- **Python 3** and associated libraries (TensorFlow/PyTorch, OpenCV, etc.)
- **CMake** and build tools for ROS2

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
