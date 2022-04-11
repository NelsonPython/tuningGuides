## Introduction

This guide is targeted towards users who are already familiar with Intel&reg; Distribution of OpenVINO&trade; toolkit and provides pointers and system setting for hardware and software that will provide the best performance for most situations. However, please note that we rely on the users to carefully consider these settings for their specific scenarios, since Intel&reg; Distribution of OpenVINO&trade; toolkit can be deployed in multiple ways and this is a reference to one such use-case.
OpenVINO&trade; toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel&reg; hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.
OpenVINO&trade; toolkit:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across an Intel&reg; CPU, Intel&reg; Integrated Graphics, Intel&reg; Neural Compute Stick 2 and Intel&reg; Vision Accelerator Design with Intel&reg; Movidius&trade; VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards, including OpenCV* and OpenCL&trade;

3rd Gen Intel Xeon Scalable processors deliver industry-leading, workload-optimized platforms with built-in AI acceleration, providing a seamless performance foundation to help speed data&rsquo;s transformative impact, from the multi-cloud to the intelligent edge and back.

### OpenVINO Toolkit Workflow

The following diagram illustrates the typical OpenVINO workflow:

<img alt="Typical OpenVINO™ workflow" height="800" src="/content/dam/develop/external/us/en/images/openvino-flow.jpg" width="539"/>

Figure 1. Typical OpenVINO workflow<a href="https://docs.openvinotoolkit.org/latest/index.html#openvino_toolkit_components">[1]

### OpenVINO Toolkit Components

Intel&reg; Distribution of OpenVINO toolkit includes the following components:

- Deep Learning Model Optimizer:  a cross-platform command-line tool for importing models and preparing them for optimal execution with the Inference Engine. The Model Optimizer imports, converts, and optimizes models, which were trained in popular frameworks, such as Caffe*, TensorFlow*, MXNet*, Kaldi*, and ONNX*.

- Deep Learning Inference Engine: a unified API to allow high performance inference on many hardware types including Intel&reg; CPU, Intel&reg; Integrated Graphics, Intel&reg; Neural Compute Stick 2, Intel&reg; Vision Accelerator Design with Intel&reg; Movidius vision processing unit (VPU).

- Inference Engine Samples: a set of simple console applications demonstrating how to use the Inference Engine in your applications.

- Deep Learning Workbench: a web-based graphical environment that allows you to easily use various sophisticated OpenVINO toolkit components.

- Post-Training Optimization tool: a tool to calibrate a model and then execute it in the INT8 precision.

- Additional Tools:
  - Benchmark App
  - Cross Check Tool
  - Compile tool

- Open Model Zoo
  - Demos: console applications that provide robust application templates to help you implement specific deep learning scenarios.
  - Additional Tools to work with models:
	  - Accuracy Checker Utility
	  - Model Downloader

- Documentation for Pretrained Models:  available in the Open Model Zoo repository

- Deep Learning Streamer (DL Streamer):  streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. DL Streamer can be installed by the Intel&reg; Distribution of OpenVINO toolkit installer. Its open source version is available on GitHub. For the DL Streamer documentation, see:
	- DL Streamer Samples
	- API Reference
	- Elements
	- Tutorial

- OpenCV: OpenCV* community version compiled for Intel&reg; hardware

- Intel&reg; Media SDK: Intel&reg; Distribution of OpenVINO toolkit for Linux only 

For building the Inference Engine from the source code, see the build instructions.

## Installation Guides

Please follow the steps below to install OpenVINO and configure the third-party dependencies based on your preference. Please look at the Target System Platform requirements before installation.

