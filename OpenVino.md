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
  - <a href="https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_compile_tool_README.html">Compile tool</a>

- <a href="https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html">Open Model Zoo</a>
	- <a href="https://docs.openvinotoolkit.org/latest/omz_demos.html">Demos</a> - Console applications that provide robust application templates to help you implement specific deep learning scenarios.
	- Additional Tools - A set of tools to work with your models including:
		- <a href="https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker.html">Accuracy Checker Utility</a>
		- <a href="https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html">Model Downloader</a>
	- <a href="https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html">Documentation for Pretrained Models</a> - Documentation for pretrained models that are available in the <a href="https://github.com/opencv/open_model_zoo">Open Model Zoo repository</a>.

- Deep Learning Streamer (DL Streamer) &ndash; Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. DL Streamer can be installed by the Intel&reg; Distribution of OpenVINO&trade; toolkit installer. Its open source version is available on <a href="https://github.com/opencv/gst-video-analytics">GitHub</a>. For the DL Streamer documentation, see:
		- <a href="https://docs.openvinotoolkit.org/latest/gst_samples_README.html">DL Streamer Samples</a>
		- <a href="https://openvinotoolkit.github.io/dlstreamer_gst/">API Reference</a>
		- <a href="https://github.com/opencv/gst-video-analytics/wiki/Elements">Elements</a>
		- <a href="https://github.com/opencv/gst-video-analytics/wiki/DL%20Streamer%20Tutorial">Tutorial</a>

- <a href="https://docs.opencv.org/master/">OpenCV</a> - OpenCV* community version compiled for Intel&reg; hardware

- <a href="/content/www/us/en/developer/tools/media-sdk/overview.html">Intel&reg; Media SDK</a> in <a href="https://docs.openvinotoolkit.org/2021.1/index.html">Intel&reg; Distribution of OpenVINO&trade; toolkit for Linux only</a>

For building the Inference Engine from the source code, see the <a href="https://github.com/openvinotoolkit/openvino/wiki/BuildingCode">build instructions</a>.

## <a id="_Toc79259922"></a>Installation Guides<a id="inpage-nav-1"></a>

Please follow the steps below to install OpenVINO&trade; and configure the third-party dependencies based on your preference. Please look at the <a href="/content/www/us/en/develop/tools/openvino-toolkit/system-requirements.html">Target System Platform requirements</a> before installation.

#### OS Based: 

Download Page: <a href="https://software.intel.com/en-us/openvino-toolkit/choose-download">https://software.intel.com/en-us/openvino-toolkit/choose-download</a>

- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html">Linux</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html">Windows</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_macos.html">macOS</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html">Raspbian OS</a>

#### Install from Images or Repositories:

- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html">Docker</a>
- <a href="https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install_from_Docker_Hub.html">Docker with DL Workbench</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html">APT</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_yum.html">YUM</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_conda.html">Anaconda Cloud</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_yocto.html">Yocto</a>
- <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html">PyPI</a>


<h2>Get Started with OpenVINO&trade; Model Zoo<a id="inpage-nav-2"></a></h2>

The Open Model Zoo is part of Intel&reg; Distribution of OpenVINO&trade; toolkit which includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. You can use these free pre-trained models instead of training your own models to speed-up the development and production deployment process.

To check the currently available models, you can use <a href="https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html">Model Downloader</a> as a very handy tool. It is a set of python scripts that can help you browse and download these pre-trained models. Other automation tools also available to leverage:

- downloader.py (model downloader) downloads model files from online sources and, if necessary, patches them to make them more usable with Model Optimizer.
- converter.py (model converter) converts the models that are not in the Inference Engine IR format into that format using Model Optimizer.
- quantizer.py (model quantizer) quantizes full-precision models in the IR format into low-precision versions using Post-Training Optimization Toolkit.
- info_dumper.py (model information dumper) prints information about the models in a stable machine-readable format

You can run the downloader.py as shown below. Note that the following example is conducted on a Linux* machine with source installation. If you plan to use it on a different setting, please change the path of the tools accordingly.

``` 
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --help

usage: downloader.py [-h] [--name PAT[,PAT...]] [--list FILE.LST] [--all]

[--print_all] [--precisions PREC[,PREC...]] [-o DIR]
[--cache_dir DIR] [--num_attempts N]
[--progress_format {text,json}] [-j N]

optional arguments:
--help show this help message and exit
--name PAT[,PAT...] download only models whose names match at least one ofspecified patterns
--list FILE.LST download only models whose names match at least one ofpatterns in the specified file
--all  download all available models
--print_all print all available models
--precisions PREC[,PREC...]

download only models with the specified precisions:
(actual for DLDT networks)

-o DIR, --output_dir DIR

path for saving models:
--cache_dir DIR directory to use as a cache for downloaded files
--num_attempts N attempt each download up to N times
--progress_format {text,json}

which format to use for progress reporting:

-j N, --jobs N how many downloads to perform concurrently</code>
```

You can use the parameter --print_all to see which pre-trained models are supported by the current version of OpenVINO for download. We will choose a classical computer vision network to detect the target picture. With the command below, we will download ssd_mobilenet_v1_coco using <a href="https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html">Model Downloader</a>.

``` 
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name ssd_mobilenet_v1_coco</code>
```

## OpenVINO&trade; Model Optimizer<a id="inpage-nav-3"></a>

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.  For more information: <a href="https://docs.openvinotoolkit.org/2021.3/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">Model Optimizer Development Guide</a>

Model Optimizer process assumes you have a network model trained using a supported deep learning framework. The scheme below illustrates the typical workflow for deploying a trained deep learning model:

<img alt="" height="218" src="/content/dam/develop/external/us/en/images/openvino-workflow.jpg" width="750"/>

Figure 2. Typical workflow for deploying a trained deep learning model<a href="https://docs.openvinotoolkit.org/2021.3/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">[2]</a>

Network topology files are described in .xml format.  Weights and biases are stored in binary format in .bin files.

To be able to convert ssd_mobilenet_v1_coco model into IR, some model specific parameters are needed to be provided the Model Optimizer. Since we downloaded this model from Open Model Zoo, we also have created a yml file to provide model specific information in each file. Here is an example for ssd_mobilenet_v1_coco:

``` 
cat /opt/intel/openvino_2021/deployment_tools/open_model_zoo/models/public/ssd_mobilenet_v1_coco/model.yml
```

The Model Downloader also contains another handy script &#39;converter.py&#39; that helps us to accurately input the parameters of the downloaded model to the Model Optimizer (MO). We can use this script directly for model conversion and reduce the workload considerably.

``` 
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/converter.py \
--download_dir=. \
--output_dir=. \
--name=ssd_mobilenet_v1_coco \
--dry_run
```

We can either let &ldquo;converter.py&rdquo; convert the model directly or use the MO execution parameters that are generated by the command above and use it when running MO.

``` 
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--framework=tf \
--data_type=FP32 \
--output_dir=public/ssd_mobilenet_v1_coco/FP32 \
--model_name=ssd_mobilenet_v1_coco \
--reverse_input_channels \
--input_shape=[1,300,300,3] \
--input=image_tensor \
--output=detection_scores,detection_boxes,num_detections \
--transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config=public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config \
--input_model=public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```


Model Optimizer arguments:

Common parameters:

``` 
- Path to the Input Model:

/root/jupyter_root/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.

pb

- Path for generated IR: /root/jupyter_root/public/ssd_mobilenet_v1_coco/FP32
- IR output name: ssd_mobilenet_v1_coco
- Log level: ERROR
- Batch: Not specified, inherited from the model
- Input layers: image_tensor
- Output layers: detection_scores,detection_boxes,num_detections
- Input shapes: [1,300,300,3]
- Mean values: Not specified
- Scale values: Not specified
- Scale factor: Not specified
- Precision of IR: FP32
- Enable fusing: True
- Enable grouped convolutions fusing: True
- Move mean values to preprocess section: None
- Reverse input channels: True


TensorFlow specific parameters:

- Input model in text protobuf format: False
- Path to model dump for TensorBoard: None
- List of shared libraries with TensorFlow custom layers implementation: None
- Update the configuration file with input/output node names: None
- Use configuration file used to generate the model with Object Detection API: /root/jupyter_root/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config
- Use the config file: None
- Inference Engine found in: /opt/intel/openvino/python/python3.6/openvino

Inference Engine version: 2.1.2021.3.0-2787-60059f2c755-releases/2021/3

Model Optimizer version: 2021.3.0-2787-60059f2c755-releases/2021/3Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /root/jupyter_root/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml
[ SUCCESS ] BIN file: /root/jupyter_root/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.bin
[ SUCCESS ] Total execution time: 47.92 seconds.
[ SUCCESS ] Memory consumed: 455 MB.</code></pre>


## Practice Inference Engine API<a id="inpage-nav-4"></a>

After creating Intermediate Representation (IR) files using the Model Optimizer, use the Inference Engine to infer the result for a given input data. The Inference Engine is a 2C++ library with a set of C++ classes to infer input data (images) and get a result. The C++ library provides an API to read the Intermediate Representation, set the input and output formats, and execute the model on devices.

Inference Engine uses a plugin architecture. Inference Engine plugin is a software component that contains complete implementation for inference on a certain Intel&reg; hardware device: CPU, GPU, VPU, FPGA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs. Integration process consists of the following steps:

<img alt="Integration process" height="313" src="/content/dam/develop/external/us/en/images/openvino-integration.jpg" width="950"/>

Figure 3. Integration process <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html">[3]</a>

### Load Plugin<a id="inpage-nav-4-undefined"></a>

Create Inference Engine Core to manage available devices and their plugins internally.

<img alt="Load Plugin" height="155" src="/content/dam/develop/external/us/en/images/openvino-code1.jpg" width="950"/>

### Read Model IR<a id="inpage-nav-4-1"></a>

Read a model IR created by the Model Optimizer.

<img alt="Real Model IR" height="260" src="/content/dam/develop/external/us/en/images/openvino-code2.jpg" width="950"/>

### Configure Input &amp; Output<a id="inpage-nav-4-2"></a>

The information about the input and output layers of the network is stored in the loaded neural network object <strong>net</strong>, and we need to obtain the information about the input and output layers and set the inference execution accuracy of the network by the following two parameters.

 - input_info
 - outputs

<img alt="Configure Input &amp; Output" height="327" src="/content/dam/develop/external/us/en/images/openvino-code3.jpg" width="950"/>

### Load Model<a id="inpage-nav-4-3"></a>

Load the model to the device using

 - InferenceEngine::Core::LoadNetwork()

<img alt="Load Model" height="60" src="/content/dam/develop/external/us/en/images/openvino-code4.jpg" width="950"/>

### Create Inference Request and Prepare Input<a id="inpage-nav-4-4"></a>

To perform a neural network inference, we need to read the image from disk and bind it to the input blob. After loading the image, we need to determine the image size and layout format. For example, the default layout format of OpenCV is <strong>CHW</strong>, but the original layout of the image is <strong>HWC</strong>, so we need to modify the layout format and add the Batch size <strong>N dimension</strong>, then organize the image format according to NCHW for inferencing and resize the input image to the network input size.

<img alt=" " height="453" src="/content/dam/develop/external/us/en/images/openvino-code5.jpg" width="950"/>

### Inference Calls<a id="inpage-nav-4-5"></a>

In this tutorial, here we use the synchronous API to demonstrate how to perform inference, calling:

- InferenceEngine::InferRequest::Infer()

If we want to improve the inference performance, we can also use the asynchronous API for inference as follows:

- InferenceEngine::InferRequest::StartAsync()
- InferenceEngine::InferRequest::Wait()

<img alt="Inference Call" height="86" src="/content/dam/develop/external/us/en/images/openvino-code6.jpg" width="950"/>

### Process the Output<a id="inpage-nav-4-6"></a>

After the inference engine inputs the graph and performs inference, a result is generated. The result contains a list of classes (class_id), confidence and bounding boxes. For each bounding box, the coordinates are given relative to the upper left and lower right corners of the original image. The correspondence between class_id and the labels file allow us to parse the text corresponding to the class, which is used to facilitate human reading comprehension.

<img alt="" height="1111" src="/content/dam/develop/external/us/en/images/openvino-code7.jpg" width="950"/>

### Visualization of the Inference Results

### <a id="inpage-nav-4-7"></a>

<img alt="" height="134" src="/content/dam/develop/external/us/en/images/openvino-code8.jpg" width="950"/>

###  

<a id="_MON_1686429488"></a>

<img alt="Open Vino Horse" height="375" src="/content/dam/develop/external/us/en/images/openvino-horse.jpg" width="500"/>

Figure 4. Image example

## <a id="_Toc79259934"></a>Practice Post-Training Optimization Tool<a id="inpage-nav-5"></a>

Post-training Optimization Tool (POT) is designed to accelerate the inference of deep learning models by applying special methods without model retraining or fine-tuning, liked post-training quantization. Therefore, the tool does not require a training dataset or a pipeline. To apply post-training algorithms from the POT, you need:

- A full precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format
- A representative calibration dataset of data samples representing a use case scenario, for example, 300 images

The tool is aimed to fully automate the model transformation process without changing the model structure. The POT is available only in the Intel&reg; Distribution of OpenVINO&trade; toolkit and is not open sourced. For details about the low-precision flow in OpenVINO&trade;, see the <a href="https://docs.openvinotoolkit.org/latest/pot_docs_LowPrecisionOptimizationGuide.html">Low Precision Optimization Guide</a>.

Post-training Optimization Tool includes a standalone command-line tool and a Python* API that provide the following key features:

- Two post-training 8-bit quantization algorithms: fast <a href="https://docs.openvinotoolkit.org/latest/pot_compression_algorithms_quantization_default_README.html">DefaultQuantization</a> and precise <a href="https://docs.openvinotoolkit.org/latest/pot_compression_algorithms_quantization_accuracy_aware_README.html">AccuracyAwareQuantization</a>.

- Global optimization of post-training quantization parameters using the <a href="https://docs.openvinotoolkit.org/latest/pot_compression_optimization_tpe_README.html">Tree-Structured Parzen Estimator</a>.

- Symmetric and asymmetric quantization schemes. For details, see the <a href="https://docs.openvinotoolkit.org/latest/pot_compression_algorithms_quantization_README.html">Quantization</a> section.

- Compression for different hardware targets such as CPU and GPU.

- Per-channel quantization for Convolutional and Fully-Connected layers.

- Multiple domains: Computer Vision, Recommendation Systems.

- Ability to implement a custom optimization pipeline via the supported <a href="https://docs.openvinotoolkit.org/latest/pot_compression_api_README.html">API</a>.

Before we start using the POT tool, we will need to prepare some config files:

- dataset files

- dataset definitions file: dataset_definitions.yml

- model json config for POT: ssd_mobilenetv1_int8.json

- model accuracy checker config: ssd_mobilenet_v1_coco.yml


### <a id="_Toc79259935"></a>Dataset Preparation<a id="inpage-nav-5-undefined"></a>

In this tutorial, we are using the dataset of <a href="https://cocodataset.org/#home">Common Objects in Context (COCO)</a> which the model was trained with this dataset. Please prepare the dataset according to <a href="https://github.com/openvinotoolkit/open_model_zoo/blob/release/data/datasets.md">Dataset Preparation Guide</a>.

To download COCO dataset, you need to follow the steps below:

- Download <a href="http://images.cocodataset.org/zips/val2017.zip">2017 Val images</a> and <a href="http://images.cocodataset.org/annotations/annotations_trainval2017.zip">2017 Train/Val annotations</a>

- Unpack archives

### <a id="_Toc79259936"></a>Global Dataset Configuration<a id="inpage-nav-5-1"></a>

If you want use definitions file in quantization via Post Training Optimization Toolkit (POT), you need to input the correct file path in these fields in the global dataset configuration file:

- annotation_file: [PATH_TO_DATASET]/instances_val2017.json
- data_source: [PATH_TO_DATASET]/val2017

### <a id="_Toc79259937"></a>Prepare Model Quantization and Configuration<a id="inpage-nav-5-2"></a>

We will need to create two config files to include model specific and dataset specific configurations to POT tool.

- ssd_mobilenetv1_int8.json

- ssd_mobilenet_v1_coco.yml


1. Create a new file and name it ssd_mobilenetv1_int8.json. This is the POT configuration file.

	<img alt="" height="524" src="/content/dam/develop/external/us/en/images/openvino-code9.jpg" width="950"/>

2. Create a dataset config file and name it ssd_mobilenet_v1_coco.yml


<img alt="" height="376" src="/content/dam/develop/external/us/en/images/openvino-code10.jpg" width="950"/>

### Quantize the Model<a id="inpage-nav-5-3"></a>

Now run the Accuracy checker tool and POT tool to create your quantized IR files.

<img alt="" height="92" src="/content/dam/develop/external/us/en/images/openvino-code11.jpg" width="950"/>

### Compare FP32 and INT8 Model Performance<a id="inpage-nav-5-4"></a>

This topic demonstrates how to run the Benchmark Python* Tool, which performs inference using convolutional networks. Performance can be measured for two inference modes: synchronous (latency-oriented) and asynchronous (throughput-oriented).

Upon start-up, the application reads command-line parameters and loads a network and images/binary files to the Inference Engine plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend on the mode defined with the -api command-line parameter. <sup>(</sup><a href="https://docs.openvinotoolkit.org/2020.4/openvino_inference_engine_tools_benchmark_tool_README.html"><sup>https://docs.openvinotoolkit.org/2020.4/openvino_inference_engine_tools_benchmark_tool_README.html</sup></a><sup>)</sup>

Please run both of your FP32 and INT8 models on <a href="https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_benchmark_tool_README.html">Benchmark Python* Tool</a> and compare your results.

<img alt="" height="84" src="/content/dam/develop/external/us/en/images/openvino-code12.jpg" width="950"/>

Now that you have run both your FP32 and INT8 IRs, you can make a comparison of the performance gain you are achieving with INT8 IR files. See the official <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_performance_benchmarks_openvino.html">benchmark results for Intel&reg; Distribution of OpenVINO&trade; Toolkit</a> on various Intel&reg; hardware settings.

## <a id="_Toc68091356"></a><a id="_Toc79259940"></a>Conclusion<a id="inpage-nav-6"></a>

This article describes an overview of Intel&reg; Distribution of OpenVINO&trade; Toolkit with how to get started guides and using the power of vector neural network instructions (VNNI) and Intel&reg; Advanced Vector Extensions (AVX512) with low precision inference workloads. The steps and codes shown are aimed at modifying the similar type of workloads to help you leverage these tutorials. You can also clearly see the performance boost of using these methodologies on <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_performance_benchmarks_openvino.html">this official benchmark result page</a>.

## <a id="_Toc79259941"></a>Additional Information<a id="inpage-nav-7"></a>

- <a href="https://devcloud.intel.com/edge/get_started/tutorials">Jupyter* Notebook Tutorials</a> - sample application Jupyter* Notebook tutorials

- <a href="https://software.intel.com/openvino-toolkit">Intel&reg; Distribution of OpenVINO&trade; toolkit Main Page</a> - learn more about the tools and use of the Intel&reg; Distribution of OpenVINO&trade; toolkit for implementing inference on the edge


## <a id="_Toc79259942"></a>References<a id="inpage-nav-8"></a>

[1] Typical OpenVINO&trade; workflow from <a href="https://docs.openvinotoolkit.org/latest/index.html#openvino_toolkit_components">https://docs.openvinotoolkit.org/latest/index.html#openvino_toolkit_components</a> on 8/4/21

[2] Typical workflow for deploying a trained deep learning model from <a href="https://docs.openvinotoolkit.org/2021.3/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">https://docs.openvinotoolkit.org/2021.3/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html</a> on 8/5/21

[3] <em>Integration process </em>from <a href="https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html%20on%208/5/21">https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html on 8/5/21</a>

