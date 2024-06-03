# Hidden Intention Discovery (HID)

PyTorch implementation of the article Uncovering the Unseen: Discover Hidden Intentions by Micro-Behavior Graph Reasoning. This work uses [`slowfast`](https://github.com/facebookresearch/SlowFast) as a baseline.

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

HID (Hidden Intention Discovery) focuses on discovering hidden intentions when humans try to hide their intentions for abnormal behavior. HID presents a unique challenge in that hidden intentions lack the obvious visual representations to distinguish them from normal intentions. We find that the difference between hidden and normal intentions can be reasoned from multiple micro-behaviors, such as gaze, attention, and facial expressions. Therefore, we first discover the relationship between micro-behaviors and hidden intentions and use multiple micro-behaviors to discover hidden intentions.

## Framework
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/51cloud/HID/blob/main/image/framework.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/51cloud/HID/blob/main/image/framework.png">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://github.com/51cloud/HID/blob/main/image/framework.png">
</picture>

## Requirements
We provide a large set of baseline results and trained models available for download in the Slowfast [Model Zoo](MODEL_ZOO.md).

Please find installation instructions for HID in [INSTALL.md](INSTALL.md). 

We will publish the HID dataset as soon.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with HID.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

## Citing PySlowFast
If you find HID useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@inproceedings{ZhouLXWZ23,
  author       = {Zhuo Zhou and
                  Wenxuan Liu and
                  Danni Xu and
                  Zheng Wang and
                  Jian Zhao},
  title        = {Uncovering the Unseen: Discover Hidden Intentions by Micro-Behavior
                  Graph Reasoning},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Multimedia,
                  {MM} 2023, Ottawa, ON, Canada, 29 October 2023- 3 November 2023},
  pages        = {6623--6633},
  publisher    = {{ACM}},
  year         = {2023},
}
