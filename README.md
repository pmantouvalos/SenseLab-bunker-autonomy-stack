# Bunker Mini 2.0 - Autonomy Stack (Work in Progress)

![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)
![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)
![OS](https://img.shields.io/badge/OS-Ubuntu%2024.04%20(WSL2)-brightgreen)

## Project Overview

This repository is currently under development as part of an academic research project at the **Technical University of Crete (SenseLab, directed by Prof. Panagiotis Partsinevelos)**. 

The goal of this project is to build a from-scratch, modular ROS 2 autonomy stack for the **AgileX Bunker Mini 2.0** tracked UGV. The final system aims to achieve autonomous exploration, resilient obstacle avoidance, and 3D artifact localization in GPS-denied, completely dark subterranean environments.

The architecture is heavily inspired by the algorithms developed during the DARPA Subterranean (SubT) Challenge, specifically focusing on lightweight solutions suitable for single-UGV deployment.

## Development Roadmap (To-Do)

- [x] Initial ROS 2 Jazzy workspace setup
- [x] Gazebo simulation environment configuration (DARPA tunnel worlds & Robot URDF)
- [ ] **Node 1:** 3D LiDAR Height-Slicing & Polar Clearance Filter
- [ ] **Node 2:** Local Motion Controller (Motion Primitives / Tentacle evaluation)
- [ ] **Node 3:** Topological Graph Explorer (Frontier detection & Dijkstra Return-to-Home)
- [ ] **Node 4:** 3D Artifact Localizer (YOLOv8 + Intel RealSense RGB-D fusion)
- [ ] Hardware deployment and physical world testing

## Theoretical Background & Literature

The upcoming implementations in this repository are based on the following research papers:
* **Local Planning:** Zhang, Ji, et al. *"Falco: Fast Likelihood-based Collision Avoidance..."* JFR, 2020.
* **Topological Exploration:** Dang, Tung, et al. *"Graph-based subterranean exploration path planning..."* JFR, 2020.
* **Artifact Detection:** Scherer, Sebastian, et al. *"Resilient and Modular Subterranean Exploration..."* Field Robotics, 2022.

---
**Author:** Panagiotis Mantouvalos | *Undergraduate Research Assistant in Robotics*  
**Laboratory:** SenseLab (Director: Prof. Panagiotis Partsinevelos)  
**Institution:** Technical University of Crete (TUC), School of Electrical and Computer Engineering (ECE)
