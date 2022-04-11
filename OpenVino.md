## Introduction

This guide is targeted towards users who are already familiar with Intel&reg; Distribution of OpenVINO&trade; toolkit and provides pointers and system setting for hardware and software that will provide the best performance for most situations. However, please note that we rely on the users to carefully consider these settings for their specific scenarios, since Intel&reg; Distribution of OpenVINO&trade; toolkit can be deployed in multiple ways and this is a reference to one such use-case.

OpenVINO&trade; toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel&reg; hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

OpenVINO&trade; toolkit:

- Enables CNN-based deep learning inference on the edge

- Supports heterogeneous execution across an Intel&reg; CPU, Intel&reg; Integrated Graphics, Intel&reg; Neural Compute Stick 2 and Intel&reg; Vision Accelerator Design with Intel&reg; Movidius&trade; VPUs

- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels

- Includes optimized calls for computer vision standards, including OpenCV* and OpenCL&trade;

<strong>3rd Gen Intel</strong><sup>&reg;</sup><strong> Xeon</strong><sup>&reg;</sup><strong> Scalable processors</strong> deliver industry-leading, workload-optimized platforms with built-in AI acceleration, providing a seamless performance foundation to help speed data&rsquo;s transformative impact, from the multi-cloud to the intelligent edge and back.

### <a id="_Toc79259920"></a>OpenVINO&trade; Toolkit Workflow<a id="inpage-nav-undefined-undefined"></a>

The following diagram illustrates the typical OpenVINO&trade; workflow:

<img alt="Typical OpenVINO™ workflow" height="800" src="/content/dam/develop/external/us/en/images/openvino-flow.jpg" width="539"/>

Figure 1. Typical OpenVINO&trade; workflow<a href="https://docs.openvinotoolkit.org/latest/index.html#openvino_toolkit_components">[1]</a>


### OpenVINO&trade; Toolkit Components<a id="inpage-nav-undefined-1"></a>

Intel&reg; Distribution of OpenVINO&trade; toolkit includes the following components:

- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">Deep Learning Model Optimizer</a> - A cross-platform command-line tool for importing models and preparing them for optimal execution with the Inference Engine. The Model Optimizer imports, converts, and optimizes models, which were trained in popular frameworks, such as Caffe*, TensorFlow*, MXNet*, Kaldi*, and ONNX*.

- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html">Deep Learning Inference Engine</a> - A unified API to allow high performance inference on many hardware types including Intel&reg; CPU, Intel&reg; Integrated Graphics, Intel&reg; Neural Compute Stick 2, Intel&reg; Vision Accelerator Design with Intel&reg; Movidius&trade; vision processing unit (VPU).

- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Samples_Overview.html">Inference Engine Samples</a> - A set of simple console applications demonstrating how to use the Inference Engine in your applications.

- <a href="https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Introduction.html">Deep Learning Workbench</a> - A web-based graphical environment that allows you to easily use various sophisticated OpenVINO&trade; toolkit components.

- <a href="https://docs.openvinotoolkit.org/latest/pot_README.html">Post-Training Optimization tool</a> - A tool to calibrate a model and then execute it in the INT8 precision.

- Additional Tools - A set of tools to work with your models including:
  - <a href="https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_benchmark_tool_README.html">Benchmark App</a>
  - <a href="https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_cross_check_tool_README.html">Cross Check Tool</a>
  - <a href="https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_compile_tool_README.html">Compile tool</a>.

- <a href="https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html">Open Model Zoo</a>
	- <a href="https://docs.openvinotoolkit.org/latest/omz_demos.html">Demos</a> - Console applications that provide robust application templates to help you implement specific deep learning scenarios.
	- Additional Tools - A set of tools to work with your models including <a href="https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker.html">Accuracy Checker Utility</a> and <a href="https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html">Model Downloader</a>.
	- <a href="https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html">Documentation for Pretrained Models</a> - Documentation for pretrained models that are available in the <a href="https://github.com/opencv/open_model_zoo">Open Model Zoo repository</a>.

- Deep Learning Streamer (DL Streamer) &ndash; Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. DL Streamer can be installed by the Intel&reg; Distribution of OpenVINO&trade; toolkit installer. Its open source version is available on <a href="https://github.com/opencv/gst-video-analytics">GitHub</a>. For the DL Streamer documentation, see:
		- <a href="https://docs.openvinotoolkit.org/latest/gst_samples_README.html">DL Streamer Samples</a>
		- <a href="https://openvinotoolkit.github.io/dlstreamer_gst/">API Reference</a>
		- <a href="https://github.com/opencv/gst-video-analytics/wiki/Elements">Elements</a>
		- <a href="https://github.com/opencv/gst-video-analytics/wiki/DL%20Streamer%20Tutorial">Tutorial</a>

- <a href="https://docs.opencv.org/master/">OpenCV</a> - OpenCV* community version compiled for Intel&reg; hardware

- <a href="/content/www/us/en/developer/tools/media-sdk/overview.html">Intel&reg; Media SDK</a> (in Intel&reg; Distribution of OpenVINO&trade; toolkit for Linux only) (<a href="https://docs.openvinotoolkit.org/2021.1/index.html">https://docs.openvinotoolkit.org/2021.1/index.html</a>)

For building the Inference Engine from the source code, see the <a href="https://github.com/openvinotoolkit/openvino/wiki/BuildingCode">build instructions</a>.

## <a id="_Toc79259922"></a>Installation Guides<a id="inpage-nav-1"></a>

Please follow the steps below to install OpenVINO&trade; and configure the third-party dependencies based on your preference. Please look at the <a href="/content/www/us/en/develop/tools/openvino-toolkit/system-requirements.html">Target System Platform requirements</a> before installation.

<table border="1" cellpadding="1" cellspacing="1" style="width:500px">

	<tbody>

		<tr>

			<td>OS Based</td>

			<td>Install from Images and Repositories</td>

		</tr>

		<tr>

			<td>

Download Page: <a href="https://software.intel.com/en-us/openvino-toolkit/choose-download">https://software.intel.com/en-us/openvino-toolkit/choose-download</a>

        - <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html">Linux</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html">Windows</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_macos.html">macOS</a>

			<a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html">Raspbian OS</a>

			</td>

			<td>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html">Docker</a>

				- <a href="https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install_from_Docker_Hub.html">Docker with DL Workbench</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html">APT</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_yum.html">YUM</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_conda.html">Anaconda Cloud</a>

				- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_yocto.html">Yocto</a>

			<a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html">PyPI</a>

			</td>

		</tr>

	</tbody>

</table>



<h2>Get Started with OpenVINO&trade; Model Zoo<a id="inpage-nav-2"></a></h2>

The Open Model Zoo is part of Intel&reg; Distribution of OpenVINO&trade; toolkit which includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. You can use these free pre-trained models instead of training your own models to speed-up the development and production deployment process.

To check the currently available models, you can use <a href="https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html">Model Downloader</a> as a very handy tool. It is a set of python scripts that can help you browse and download these pre-trained models. Other automation tools also available to leverage:

	- downloader.py (model downloader) downloads model files from online sources and, if necessary, patches them to make them more usable with Model Optimizer.

	- converter.py (model converter) converts the models that are not in the Inference Engine IR format into that format using Model Optimizer.

	- quantizer.py (model quantizer) quantizes full-precision models in the IR format into low-precision versions using Post-Training Optimization Toolkit.

	- info_dumper.py (model information dumper) prints information about the models in a stable machine-readable format

