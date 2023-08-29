# HID

PyTorch implementation of the article Uncovering the Unseen: Discover Hidden Intentions by Micro-Behavior Graph Reasoning.

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

HID (Hidden Intention Discovery) focuses on discovering hidden intentions when humans try to hide their intentions for abnormal behavior. HID presents a unique challenge in that hidden intentions lack the obvious visual representations to distinguish them from normal intentions. We find that the difference between hidden and normal intentions can be reasoned from multiple microbehaviors, such as gaze, attention, and facial expressions. Therefore, we first discover the relationship between micro-behavior and hidden intentions and use graph structure to reason about hidden intentions.

## Framework
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/51cloud/HID/blob/main/image/framework.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/51cloud/HID/blob/main/image/framework.png">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://github.com/51cloud/HID/blob/main/image/framework.png">
</picture>

## Requirements
We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with PySlowFast.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

