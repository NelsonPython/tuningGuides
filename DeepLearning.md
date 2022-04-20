## Overview

This user guide is intended to explain how the 3 Generation Intel&reg; Xeon&reg; Scalable Processor platform (codename Ice Lake/Whitley) is used for machine learning and deep learning-related tasks. Executing machine learning and deep learning workloads on the Intel&reg; Xeon&reg; Scalable Processor platform has the following advantages:


- The platform is very suitable for processing memory-intensive workloads and 3D-CNN topologies used in medical imaging, GAN, seismic analysis, genome sequencing, etc.
- The simple numactl command can be used for flexible core control; it is still very suitable for real-time inference even when the number of batches is small.
- It is supported by a powerful ecosystem and can be used for distributed training (such as for computations directly at the data source) directly on large-scale clusters. This avoids the additional costs for large storage capacity and expensive cache mechanisms that are usually required for the training of scaled architecture.
- Multiple types of workloads (HPC/BigData/AI) are supported on the same cluster to achieve better TCO.
- It satisfies the computing requirements in many real deep learning applications via SIMD acceleration.
- The same infrastructure can be used directly for training and inference.


The development and deployment of typical deep learning applications involve the following stages:

<img alt="DL stages" height="148" src="/content/dam/develop/external/us/en/images/dl-stages.jpg" width="560"/>

These different stages require the allocation of following resources, and choosing the right resources can greatly accelerate the efficiency of your AI services:


- Computational power
- Memory
- Storage for datasets
- Communication link between compute nodes
- Optimized software


All the processes including dataset preparation, model training, model optimization, and model deployment, can be done on an Intel&reg; Xeon&reg; Scalable Processor platform-based infrastructure which also supports machine learning/deep learning platforms for training and inference. The proposed infrastructure is shown in the figure below:

<img alt="DL infrastructure" height="457" src="/content/dam/develop/external/us/en/images/dl-infrastructure.jpg" width="1027"/>

## Introducing Intel&reg; AVX-512 and Intel&reg; Deep Learning Boost

Intel&reg; Advanced Vector Extensions 512 (Intel&reg; AVX-512) is a single instruction, multiple data (SIMD) instruction set based on x86 processors. Compared to traditional single instruction, single data instructions, a SIMD instruction allows for executing multiple data operations with a single instruction. As the name implies, Intel&reg; AVX-512 has a register width of 512 bits, and it supports 16 32-bit single-precision floating-point numbers or 64 8-bit integers.

Intel&reg; Xeon&reg; Scalable Processors support multiple types of workloads, including complex AI workloads, and improve AI computation performance with the use of Intel&reg; Deep Learning Boost (Intel&reg; DL Boost). Intel Deep Learning Boost includes Intel&reg; AVX-512 VNNI (Vector Neural Network Instructions) which is an extension to the Intel&reg; AVX-512 instruction set. It can combine three instructions into one for execution, which further unleashes the computing potential of next-generation Intel&reg; Xeon&reg; Scalable Processors and increases the inference performance of the INT8 model. Both 2nd-Generation and 3rd-Generation Intel&reg; Xeon&reg; Scalable Processors support VNNI.

<img alt="DL VNNI" height="176" src="/content/dam/develop/external/us/en/images/dl-vnni.jpg" width="855"/>

Platforms not using VNNI require the  vpmaddubsw, vpmaddwd and  vpaddd instructions to complete the multiply-accumulate operations in INT8 convolution operation:

<img alt="DL int-8" height="140" src="/content/dam/develop/external/us/en/images/dl-int8-1.jpg" width="1175"/>

Platforms using VNNI require only one instruction, &ldquo;vpdpbusd&rdquo;, to complete the INT8 convolution operation:

<img alt="DL int-8" height="156" src="/content/dam/develop/external/us/en/images/dl-int8-2.jpg" width="587"/>

## BIOS Settings and Hardware Configurations

### BIOS Settings

The configuration items that can be optimized in BIOS and their recommended values are as follows:

<table>
	<tbody>
		<tr>
			<td>
			Configuration item
			</td>
			<td>
			Recommended value
			</td>
		</tr>
		<tr>
			<td>
			Hyper-Threading
			</td>
			<td>
			Enable
			</td>
		</tr>
		<tr>
			<td>
			SNC (Sub NUMA)
			</td>
			<td>
			Disable
			</td>
		</tr>
		<tr>
			<td>
			Boot performance mode
			</td>
			<td>
			Max Performance
			</td>
		</tr>
		<tr>
			<td>
			Turbo Mode
			</td>
			<td>
			Enable
			</td>
		</tr>
		<tr>
			<td>
			Hardware P-State
			</td>
			<td>
			Native Mode
			</td>
		</tr>
	</tbody>
</table>

### Recommended Hardware Configurations

Machine learning workloads, and in particular deep learning workloads, are usually used for compute-intensive applications. Hence, they require a selection of suitable types of memory, CPU, hard drives, and other computing resources to achieve optimal performance. In summary, the following common configurations are recommended:

 Memory configuration

The utilization of all memory channels is recommended so that the bandwidth of all memory channels can be utilized.

 CPU configuration

FMA, the Intel AVX-512 acceleration module in Intel processors, is an important component in unleashing computational performance, and artificial intelligence-related workloads are usually part of compute-intensive applications. In order to achieve better computing performance, it is recommended to use the Intel Xeon&reg; Scalable Processors Gold 6 series (or above) which have two Intel AVX512 computational modules per core.

 Network configuration

If cross-node training clusters are required, then it is recommended to choose high-speed networking such as 25G/100G networks for better scalability.

 Hard drive configuration

For high IO efficiency for workloads, SSDs and drives with higher read and write speeds are recommended.

## Linux System Optimization

### OpenMP Parameter Settings

The recommended configuration for the main parameters is as follows:


- OMP_NUM_THREADS = &ldquo;number of cpu cores in container&rdquo;
- KMP_BLOCKTIME = 1 or 0 (set according to actual type of model)
- KMP_AFFINITY=granularity=fine, verbose, compact,1,0


### Number of CPU cores

The main impact of the number of CPU cores on inference performance is as follows:

&bull; When batchsize is small (in online services for instance), the increase in inference throughput gradually weakens as the number of CPU cores increases; in practice, 8-16 CPU cores is recommended for service deployment depending on the model used.

&bull; When batchsize is large (in offline services for instance), the inference throughput can increase linearly as the number of CPU cores increases; in practice, more than 20 CPU cores is recommended for service deployment.


``` # taskset -C xxx-xxx –p pid (limits the number of CPU cores used in service) ```

### Impact of NUMA Configuration

For NUMA-based servers, there is usually a 5-10% increase in performance when configuring NUMA on the same node compared to using it on different nodes.


``` #numactl -N NUMA_NODE -l command args ... (controls NUMA nodes running in service) ```

### Configuration of Linux Performance Governor

Performance: As the name suggests, efficiency is the only consideration and the CPU frequency is set to its peak to achieve the best performance.


``` # cpupower frequency-set -g performance ```

### CPU C-States Settings

CPU C-States: To reduce power consumption when the CPU is idle, the CPU can be placed in the low-power mode. There are several power modes available for each CPU which are collectively referred to as C-states or C-modes.

Disabling C-States can increase performance.


``` #cpupower idle-set -d 2,3 ```

## Using Intel&reg; Optimization for TensorFlow* Deep Learning Framework

TensorFlow* is one of the most popular deep learning frameworks used in large-scale machine learning (ML) and deep learning (DL) applications. Since 2016, Intel and Google* engineers have been working together to use Intel&reg; oneAPI Deep Neural Network Library (Intel&reg; oneDNN) to optimize TensorFlow* performance and accelerate its training and inference performance on the Intel&reg; Xeon&reg; Scalable Processor platform.

### Deploying Intel&reg; Optimization for TensorFlow* Deep Learning Framework

Reference: <a href="/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html">https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html</a>

Step 1: Install a Python3.x environment. Here is an example to illustrate how to build Python* 3.6 with Anaconda*

Reference: <a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual</a>

Download and install the latest version of Anaconda


``` 
# wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# sh Anaconda3-2020.02-Linux-x86_64.sh

# source /root/.bashrc

# conda install python=3.6 (create a Python3.6 environment)

#(base) [root@xx]# python -V

Python 3.6.10 
```

Step 2: Install the Intel optimation for TensorFlow*: intel-tensorflow.

Install the latest version (2.x)


``` 
# pip install intel-tensorflow 
```

If you need to install tensorflow1.x, we recommend installing the following version to take advantage of the performance acceleration on the 3<sup>rd</sup> Gen Intel&reg; Xeon&reg; Scalable Processor platform:


``` 
# pip install https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp36-cp36m-manylinux2010_x86_64.whl 
```


Step 3: Set run-time optimization parameters.

Reference:

<a href="https://github.com/IntelAI/models/blob/master/docs/general/tensorflow/GeneralBestPractices.md">https://github.com/IntelAI/models/blob/master/docs/general/tensorflow/GeneralBestPractices.md</a>

Usually, the following two methods are used for inference, which use different optimization settings

 Batch inference: Batch Size &gt;1, measures the number of input tensors that can be processed per second. Usually, all the physical cores in the same CPU socket can be used for batch inference to achieve the best performance.

 On-line inference (also known as real-time inference):  Batch Size=1, a measure of time needed to process one input tensor (when the batch size is 1). In real-time inference, multiple instances are run concurrently to achieve the optimal throughput.

1: Obtaining the number of physical cores in the system:

To confirm the current number of physical cores, we recommend using the following command:


``` 
# lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs 
```

In this example, we assume 8 physical cores.

2: Setting optimization parameters:

Optimization parameters are configured using the two following methods. Please choose the configuration method according to your needs.

Method 1: Configure environment parameters directly:


``` 
export OMP_NUM_THREADS=physical cores

export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

export KMP_BLOCKTIME=1

export KMP_SETTINGS=1 
```

Method 2: Add environment variables in the Python code that is running:


``` 
import os

os.environ["KMP_BLOCKTIME"] = "1"

os.environ["KMP_SETTINGS"] = "1"

os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

if FLAGS.num_intra_threads &gt; 0:

os.environ["OMP_NUM_THREADS"]= # &lt;physical cores&gt;

config = tf.ConfigProto()

config.intra_op_parallelism_threads = # &lt;physical cores&gt;

config.inter_op_parallelism_threads = 1

tf.Session(config=config) 
```

### Inferencing using Intel&reg; Optimization for TensorFlow* DL Model with FP32/INT8 support

This section mainly explains how to run the inference benchmark on ResNet50. You can refer to the following reference to inference using your machine learning/deep learning model.

Reference: <a href="https://github.com/IntelAI/models/blob/master/docs/image_recognition/tensorflow/Tutorial.md">https://github.com/IntelAI/models/blob/master/docs/image_recognition/tensorflow/Tutorial.md</a>

Taking inference benchmarking for ResNet50* as an example, FP32, BFloat16, and Int8 are supported for model inference.

Reference: <a href="https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md">https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md</a>

FP32-based model inference: 

<a href="https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions">https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions</a>

INT8-based model inference:

<a href="https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions">https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions</a>

### Training using Intel&reg; Optimization for TensorFlow* DL Model with FP32/ INT8 Support

This section mainly explains how to run a training benchmark on ResNet50. You can refer to the following reference to run your machine learning/deep learning model training.

FP32-based training:

<a href="https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-training-instructions">https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-training-instructions</a>

### Applications &ndash; Inferencing and Training Using Intel Optimized TensorFlow Wide &amp; Deep Model

Among the many operations in the data center, it is a typical application to use recommendation systems to match users with content they are interested in. Recommendation system is a type of information filtering system that learns about users&#39; interests according to their profiles and past behavior records and predict their ratings or preferences for a given item. It changes the way a business communicates with users and enhances the interaction between the business and its users.

When using deep learning, we find, from a large amount of complex raw data, the deep interactions between features that are difficult to be expressed with traditional machines using artificial feature engineering. Related study outcomes include Wide &amp; Deep, DeepFM, FNN, DCN, and other models.

Using the Wide &amp; Deep model as an example, the core idea is to take advantage of both the memorization capability of a linear model and the generalization capability of the DNN model and optimize the parameters in these models at the same time during training. This will result in better overall model prediction capabilities. Its structure is shown in the figure below:

<img alt="DL structure" height="389" src="/content/dam/develop/external/us/en/images/dl-structure.jpg" width="813"/>

 Wide

&quot;Wide&quot; is a generalized linear model, and its inputs mainly include original and interactive features. We can use cross-product transformation to build the interactive features of K-group:

<img alt="Wide" height="112" src="/content/dam/develop/external/us/en/images/dl-wide.jpg" width="426"/>

 Deep

&ldquo;Deep&rdquo; is a DNN model, and the calculation for each layer is as follows:

<img alt="Deep" height="98" src="/content/dam/develop/external/us/en/images/dl-deep.jpg" width="437"/>

 Co-training

The Wide &amp; Deep model uses co-training instead of integration. The difference is that co-training shares a loss function, then updates the parameters in either part of the model at the same time, while integration trains N models independently and fuses them together afterwards. Therefore, the output of the model is:

<img alt="co-training" height="68" src="/content/dam/develop/external/us/en/images/dl-cotrain.jpg" width="680"/>

The above is the background information on the Wide &amp; Deep model. Next, we will describe how to run inference benchmarking for the Wide &amp; Deep model.

Reference:

<a href="https://github.com/IntelAI/models/blob/master/docs/recommendation/tensorflow/Tutorial.md">https://github.com/IntelAI/models/blob/master/docs/recommendation/tensorflow/Tutorial.md</a>

Dataset preparation:

<a href="https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#Prepare-dataset">https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#Prepare-dataset</a>

FP32-based model inference:

<a href="https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#fp32-inference-instructions">https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#fp32-inference-instructions</a>

INT8-based model inference:

<a href="https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#int8-inference-instructions">https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#int8-inference-instructions</a>

FP32-based training:

<a href="https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#fp32-training-instructions">https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds#fp32-training-instructions</a>

### Intel&reg; Math Kernel Library (MKL) Threadpool-Based TensorFlow (Optional)

Starting with TensorFlow 2.3.0, a new feature has been added. You can choose Eigen Threadpool for TensorFlow multi-threading support instead of OpenMP, by using the compiling option --config=mkl_threadpool instead of --config=mkl, when compiling the Tensorflow source code.

If the user wants to try this feature with TensorFlow 1.15, they need to download the source code that has been ported and optimized by Intel and compile it (it should be particularly pointed out that  Bazel* 0.24.1 needs to be installed for the purpose):


``` 
# git clone https://github.com/Intel-tensorflow/tensorflow.git

# git checkout -b tf-1.15-maint remotes/origin/tf-1.15-maint

# bazel --output_user_root=$BUILD_DIR build --config=mkl_threadpool -c opt --copt=-O3 //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package $BUILD_DIR 
```

After successfully completing the steps above, the TensorFlow  <em>wheel</em> file can be found under the  <em>$BUILD_DIR</em> path. For example:  <em>tensorflow-1.15.0up2-cp36-cp36m-linux_x86_64.whl</em>. The installation steps are as follows:


``` 
# pip uninstall tensorflow

# pip install $BUILD_DIR/&lt;filename&gt;.whl --user 
```

## Using PyTorch*, a Deep Learning Framework

### Deploying PyTorch

Reference: <a href="/content/www/us/en/developer/articles/guide/getting-started-with-intel-optimization-of-pytorch.html">https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-intel-optimization-of-pytorch.html</a>

Environment: Python3.6 or above

Step 1: Visit the official PyTorch website: <a href="https://pytorch.org/">https://pytorch.org/</a>

Step 2: Select CPU

Currently, Intel oneDNN is integrated into the official version of PyTorch, so there is no need for additional installation to have accelerated performance on the Intel&reg; Xeon&reg; Scalable Processor platform. Select &ldquo;None&rdquo; for CUDA. See the figure below for details.

<img alt="oneDNN" height="372" src="/content/dam/develop/external/us/en/images/dl-onednn.jpg" width="1060"/>

Step 3: Installation


``` 
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 
```


### Optimization Recommendations for Training and Inferencing PyTorch-based Deep Learning Models

You may refer to the following website to learn more about optimization parameter settings for PyTorch* on the Intel&reg; Xeon&reg; Scalable Processor platform.

Reference: <a href="/content/www/us/en/developer/articles/technical/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html">https://www.intel.com/content/www/us/en/developer/articles/technical/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html</a>

### Introducing and Using Intel&reg; Extension for PyTorch

Intel&reg; Extension for PyTorch is a Python extension of PyTorch that aims to improve the computational performance of PyTorch on Intel&reg; Xeon&reg; Processors. Not only does this extension includes additional functions, but it also provides performance optimizations for new Intel hardware.

The Github links to the Intel Extension for PyTorch are:

<a href="https://github.com/intel/intel-extension-for-pytorch">https://github.com/intel/intel-extension-for-pytorch</a>

<a href="https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_Extensions_AutoMixedPrecision">https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_Extensions_AutoMixedPrecision</a>

## Accelerating Vector Recall in the Recommendation System with Intel&reg; Deep Learning Boost VNNI

A problem that needs to be resolved in the recommendation system is how to generate a recommendation list with the length of K for a given user that matches their interests and needs as much as possible (high accuracy) and as fast as possible (low latency)? Conventional recommendation systems include two components: vector recall and ranking. &ldquo;Vector recall&rdquo; roughly filters out hundreds or thousands of items from a huge recommendation pool that will most likely interest the user, passes the results on to the ranking module for further sorting before the final recommendation results are obtained.

<img alt="vector recall" height="552" src="/content/dam/develop/external/us/en/images/dl-vector-recall.jpg" width="1026"/>

Vector recall can be converted into a high-dimensional vector similarity search problem.

The Hierarchical Navigable Small World (HNSW) algorithm is a type of Approximate Nearest Neighbor (ANN) vector similarity search algorithm based on graph structures. It is also one of the fastest and most precise algorithms.

<img alt="HNSW" height="512" src="/content/dam/develop/external/us/en/images/dl-ann.jpg" width="482"/>

Usually, the data type of the raw vector data is FP32. For many applications (such as image search), vector data can be expressed in INT8/INT6 and the impact of quantization error on the final search result is limited. The &ldquo;VNNI intrinsic&rdquo; instruction can be used for inner product calculations for INT8/INT6 vectors. Many experiments have shown that QPS Performance has greatly improved, and that recall rate remains virtually unchanged. The reason for the improvement in QPS performance is that the memory&ndash;bandwidth ratio for INT8/INT16 is smaller than for FP32, and VNNI instructions accelerate the distance calculations in addition.

Currently, optimized source code is implemented based on the HNSWLib[10] open source project. We have already ported it to the Faiss[9] framework, which is widely used in the industry.

To achieve the optimal performance, the following deployment steps are recommended:

- Bind NUMA
- Each physical CPU core executes a single query process

Reference command (using 1 socket and 24 cores as an example):


``` 
# numactl -C 0-23 <test_program> 
```

When the dataset is large (in the range of 100 million to billions for example), the traditional approach is to slice the dataset into several smaller datasets to get the  topK for each dataset separately before merging them back together at the end. Since the amount of communication between multiple machines has increased, latency also increases while the QPS performance decreases. Our experience with HNSW on large datasets show that it is better not to slice datasets if possible, but rather establish indices and execute searches on complete datasets to get the best performance. When a dataset is too large and there is not enough DDR space (e.g. local memory space), you can consider using PMem (Intel&reg; Optane&trade; persistent memory)

By saving the  HNSW layer0 data on PMEM, the size of the dataset that can be supported has greatly increased (a single socket can support an INT8 database with up to 4 billion records @  d=100). The persistence feature allows you to skip the loading process for a large amount of data, which greatly reduces the time it takes to initialize.

## AI Neural Network Model Quantization

### AI neural network quantization process

Computations in neural networks are mainly concentrated in the convolutional layer and the fully connected layer. The computations on these two layers can be expressed as:  Y = X * Weights + Bias. Therefore, it is natural to focus on matrix multiplication to optimize performance. The way to begin neural network model quantization is by trading-off precision (limited) for performance improvement. By replacing 32-bit floating-point numbers with low-precision integers for matrix operations, it not only speeds up calculations, but also compresses the model, thus saving memory bandwidth.

There are three approaches to the quantization of neural network models:


- Post-Training Quantization (PTQ), which is supported by most AI frameworks.
- Quantization-Aware-Training (QAT), which inserts the  FakeQuantization node into the FP32 model when the training converges. It increases the quantization-induced noise. During the backpropagation stage of the training, the model weights fall into a finite interval which results in better quantization precision.
- Dynamic Quantization (DQ) is very similar to PTQ. They are both quantization methods used on post-trained models. The difference lies in that the quantization factor in the activation layer is dynamically decided by the data range used when the neural network model is run, while for PTQ samples from a small-scale pre-processed dataset are used to obtain data distribution and range information in the activation layer, then records it permanently in the newly generated quantization model. Of the Intel&reg; AI Quantization Tools for TensorFlow which we will talk about later on,  onnxruntime supports this method at the backend only.


The basic procedure for the post-training quantization of neural networks is as follows:

     1. Fuse FP32 OP to INT8 OP. For example, <em>MatMul</em>, <em>BiasAdd</em> and <em>ReLU</em> can be fused into a single quantized OP at the fully connected layer,  <em>QuantizedMatMulWithBiasAndRelu</em>. Different neural network frameworks support different fuse-able OPs. For Intel&reg; AI Quantization Tools for TensorFlow, which will be discussed later on, below we can see a list of fuse-able OPs supported by TensorFlow: <a href="https://github.com/intel/lpot/blob/master/lpot/adaptor/tensorflow.yaml#L190">https://github.com/intel/lpot/blob/master/lpot/adaptor/tensorflow.yaml#L190</a>.

For fuse-able OPs supported by pyTorch, please see : <a href="https://github.com/intel/lpot/blob/master/lpot/adaptor/pytorch_cpu.yaml#L124">https://github.com/intel/lpot/blob/master/lpot/adaptor/pytorch_cpu.yaml#L124</a>

     2. Quantize weights and save them in the quantized model.

     3. Quantize the input/activation layer by sampling the calibration dataset to acquire the distribution and range information of the data in the activation layer, which is then recorded in the newly generated quantized model.

     4. The  Requantize operation is fused into its corresponding INT8 OP to generate the final quantized model.

Using a simple model which includes two layers of  MatMul as an example, we can observe the quantization process as follows:

<img alt="MatMul" height="585" src="/content/dam/develop/external/us/en/images/dl-matmul.jpg" width="695"/>

### Intel&reg; AI Quantization Tools for TensorFlow

Intel&reg; AI Quantization Tools for TensorFlow is an open source Python library which provides API access for low-precision quantization for cross-neural network development frameworks. It is intended to provide simple, easy-to-use and precision-driven auto tuning tools for the quantization of models for accelerating the inference performance of low-precision models on the 3rd Gen Intel&reg; Xeon&reg; Scalable Processor platform.

Reference: <a href="https://github.com/intel/lpot">https://github.com/intel/lpot</a>

<img alt="AI tools" height="452" src="/content/dam/develop/external/us/en/images/dl-ai-tools.jpg" width="912"/>

Intel&reg; AI Quantization Tools for TensorFlow currently support the following Intel optimized deep learning frameworks:


- <a href="https://www.tensorflow.org/">Tensorflow*</a>
- <a href="https://pytorch.org/">PyTorch*</a>
- <a href="https://mxnet.apache.org/">Apache* MXNet</a>
- <a href="https://onnx.ai/">ONNX Runtime</a>


The frameworks and their versions that have already been verified are shown below:

<table>
	<tbody>
		<tr>
			<td>
			OS
			</td>
			<td>
			Python
			</td>
			<td>
			Framework
			</td>
			<td>
			Version
			</td>
		</tr>
		<tr>
			<td rowspan="10">
			CentOS 7.8

			Ubuntu 18.04
			</td>
			<td rowspan="10">
			3.6

			3.7
			</td>
			<td rowspan="6">
			TensorFlow
			</td>
			<td>
			2.2.0
			</td>
		</tr>
		<tr>
			<td>
			1.15.0 UP1
			</td>
		</tr>
		<tr>
			<td>
			1.15.0 UP2
			</td>
		</tr>
		<tr>
			<td>
			2.3.0
			</td>
		</tr>
		<tr>
			<td>
			2.1.0
			</td>
		</tr>
		<tr>
			<td>
			1.15.2
			</td>
		</tr>
		<tr>
			<td>
			PyTorch
			</td>
			<td>
			1.5.0+cpu
			</td>
		</tr>
		<tr>
			<td rowspan="2">
			Apache* MXNet
			</td>
			<td>
			1.7.0
			</td>
		</tr>
		<tr>
			<td>
			1.6.0
			</td>
		</tr>
		<tr>
			<td>
			ONNX Runtime
			</td>
			<td>
			1.6.0
			</td>
		</tr>
	</tbody>
</table>

The tuning strategies supported by Intel&reg; AI Quantization Tools for Tensorflow include:


- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#basic">Basic</a>
- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#bayesian">Bayesian</a>
- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#exhaustive">Exhaustive</a>
- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#mse">MSE</a>
- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#random">Random</a>
- <a href="https://github.com/intel/lpot/blob/master/docs/tuning_strategies.md#tpe">TPE</a>


The workflow for Intel&reg; AI Quantization Tools for TensorFlow is shown below. The model quantization parameters matching the precision loss target are automatically selected according to the set tuning strategy, and the quantized model is generated:

<img alt="TensorFlow" height="430" src="/content/dam/develop/external/us/en/images/dl-tensorflow.jpg" width="855"/>

### Installing Intel&reg; AI Quantization Tools for TensorFlow

For details on installation, refer to: <a href="https://github.com/intel/lpot/blob/master/README.md">https://github.com/intel/lpot/blob/master/README.md</a>

Step 1: Use  Anaconda to create a  Python3.x virtual environment with the name of  lpot. We are using  Python 3.7 here as an example:


```
# conda create -n lpot python=3.7

# conda activate lpot 
```

Step 2: Install  lpot; the two following installation methods are available:

Installing with the binary file:


``` 
# pip install lpot 
```

Install from the source code


``` 
# git clone https://github.com/intel/lpot.git

# cd lpot

# pip install –r requirements.txt

# python setup.py install 
```

### Using Intel&reg; AI Quantization Tools for TensorFlow

We are using  ResNet50 v1.0 as an example to explain how to use this tool for quantization.

### Dataset preparation:

Step 1: Download and decompress the ImageNet validation dataset:


``` 
# mkdir –p img_raw/val &amp;&amp; cd img_raw

# wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar

# tar –xvf ILSVRC2012_img_val.tar -C val 
```

Step 2: Move the image files to the child directories sorted by label:


``` 
# cd val

# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash 
```

Step 3: Use the script, <a href="https://github.com/intel/lpot/blob/master/examples/tensorflow/image_recognition/prepare_dataset.sh">prepare_dataset.sh</a>, to convert raw data to the  TFrecord format:


``` 
# cd examples/tensorflow/image_recognition

# bash prepare_dataset.sh --output_dir=./data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation 
```

Reference: <a href="https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition#2-prepare-dataset">https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition#2-prepare-dataset</a>

### Model preparation:

``` 
# wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb 
```

### Run Tuning:

Edit the file: <a href="https://github.com/intel/lpot/blob/master/examples/tensorflow/image_recognition/resnet50_v1.yaml">examples/tensorflow/image_recognition/resnet50_v1.yaml</a>, making sure the dataset path for  quantizationcalibration,  evaluationaccuracy and  evaluationperformance is the user&#39;s real local path. It should be where the  TFrecord data generated previously during the data preparation stage, is located.

``` 
# cd examples/tensorflow/image_recognition

# bash run_tuning.sh --config=resnet50_v1.yaml \n
--input_model=/PATH/TO/resnet50_fp32_pretrained_model.pb \n
--output_model=./lpot_resnet50_v1.pb 
```

Reference: <a href="https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition#1-resnet50-v10">https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition#1-resnet50-v10</a>

### Run Benchmark:

``` 
# bash run_benchmark.sh --input_model=./lpot_resnet50_v1.pb --config=resnet50_v1.yaml 
```

The output is shown below. The performance data is for reference only:

Accuracy mode benchmark result:

 Accuracy is 0.739

 Batch size = 32

 Latency: (results will vary)

 Throughput: (results will vary)
 

Performance mode benchmark result:

 Accuracy is 0.000

 Batch size = 32

 Latency: (results will vary)

 Throughput: (results will vary)

## Using Intel&reg; Distribution of OpenVINO&trade; Toolkit for Inference Acceleration<a id="_Hlk43400832"></a>

### Intel&reg; Distribution of OpenVINO&trade; Toolkit

Intel&reg; Distribution of OpenVINO<sup>TM </sup>toolkit&rsquo;s official website and download websites:

<a href="/content/www/us/en/developer/tools/openvino-toolkit/overview.html">https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html</a>

Online documentation:

<a href="https://docs.openvino.ai/latest/index.html">https://docs.openvino.ai/latest/index.html</a>

Online documentation in Simplified Chinese:

<a href="https://docs.openvino.ai/cn/latest/index.html">https://docs.openvino.ai/cn/latest/index.html</a>

The Intel&reg; Distribution of OpenVINO<sup>TM</sup> toolkit is used to accelerate the development of computer vision and deep learning applications. It supports deep learning applications with various accelerators, including CPUs, GPUs, FPGAs, and Intel&reg; Movidius&trade; CPUs on the Intel&reg; Xeon&reg; Processor platform, and it also directly supports heterogenous execution.

<img alt="DL OpenVINO" height="353" src="/content/dam/develop/external/us/en/images/dl-openvino.jpg" width="1020"/>

The Intel&reg; Distribution of OpenVINO<sup>TM </sup>toolkit is designed to improve the performance and reduce the development time of computer vision processing and deep learning inference solutions. It includes two components: computer vision and deep learning development kits.

The Deep Learning Deployment Toolkit (DLDT) is a cross-platform tool for accelerating deep learning inference performance, and includes the following components:


-  Model Optimizer: converts models trained with Caffe*, TensorFlow, Mxnet, and other frameworks into Intermediate Representations (IR).
-  Inference Engine: executes the IR on CPU, GPU, FPGA, VPU, and other hardware. It automatically calls the hardware acceleration kit to accelerate inference performance.


The Intel&reg; Distribution of OpenVINO<sup>TM </sup>toolkit Workflow:

<img alt="OpenVino workflow" height="372" src="/content/dam/develop/external/us/en/images/dl-openvino-workflow.jpg" width="881"/>

### Deploying the Intel&reg; Distribution of OpenVINO&trade; Toolkit

You can refer to the installation documentation in Simplified Chinese:

<a href="https://docs.openvino.ai/downloads/cn/I03030-5-Install%20Intel_%20Distribution%20of%20OpenVINO_%20toolkit%20for%20Linux%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Installing the Intel<sup>&reg;</sup> Distribution of OpenVINO&trade; toolkit for Linux*</a> :

### Using Deep Learning Deployment Toolkit (DLDT) of the Intel&reg; Distribution of OpenVINO Toolkit

<a href="https://docs.openvino.ai/downloads/cn/I03030-9-Introduction%20to%20Intel_%20Deep%20Learning%20Deployment%20Toolkit%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Introduction to the Intel<sup>&reg;</sup> Deep Learning Deployment toolkit</a>

<a id="_Hlk42691266"></a> <a href="https://docs.openvino.ai/downloads/cn/I03030-10-Image%20Classification%20Cpp%20Sample%20Async%20-%20OpenVINO_%20Toolkit.pdf">Image Classification C++ Sample (Async)</a>

<a href="https://docs.openvino.ai/downloads/cn/I03030-11-Object%20Detection%20Cpp%20Sample%20SSD%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Object Detection C++ Sample (SSD)</a>

<a href="https://docs.openvino.ai/downloads/cn/I03030-12-Automatic%20Speech%20Recognition%20Cpp%20%20Sample%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Automatic Speech Recognition C++ Sample</a>

<a id="_Hlk42691304"></a>

 <a href="https://docs.openvino.ai/downloads/cn/I03030-13-Action%20Recognition%20Python%20Demo%20-%20OpenVINO_%20Toolkit.pdf">Action Recognition Python* Demo</a>

<a href="https://docs.openvino.ai/downloads/cn/I03030-14-Crossroad%20Camera%20Cpp%20%20Demo%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Crossroad Camera C++ Demo</a>

<a href="https://docs.openvino.ai/downloads/cn/I03030-15-Human%20Pose%20Estimation%20Cpp%20Demo%20-%20OpenVINO_%20Toolkit.pdf" target="_blank"> Human Pose Estimation C++ Demo</a>

<a id="_Hlk42691333"></a> <a href="https://docs.openvino.ai/downloads/cn/I03030-16-Interactive%20Face%20Detection%20Cpp%20%20Demo%20-%20OpenVINO_%20Toolkit.pdf">Interactive Face Detection C++ Demo</a>

 

### Using the Intel&reg; Distribution of OpenVINO&trade; Toolkit for INT8 Inference Acceleration

By inferencing on an INT8-based model and using Intel DL Boost on the Intel&reg; Xeon&reg; Scalable Processor platform for acceleration, you can greatly increase inference efficiency. At the same time, it saves computing resources and reduces power consumption. The 2020 version and later versions of OpenVINO&trade; all provide INT8 quantization tools which support the quantization of FP32-based models.

The INT8-based model quantization tool provided by OpenVINO is a  Post-training  Optimization  Toolkit  (POT) is used to optimize and quantize trained models. There is no need to re-train or fine-tune models or to modify model structures. The figure below shows the process of how OpenVINO is used to optimize new models.

Step 0: Acquire the trained model,

Step 1: POT generation and optimization,

Step 2: Optional operation (Whether to fine-tune the model will be determined according to the actual situation for better accuracy), and

Step 3: Use OpenVINO IE for model inference.

<img alt="OpenVINO-ie" height="420" src="/content/dam/develop/external/us/en/images/dl-openvino-ie.jpg" width="922"/>

POT provides an independent command line tool and Python API and it mainly supports the following features:


- Two types of post-training INT8 quantization algorithms: fast <a href="https://docs.openvino.ai/latest/pot_compression_algorithms_quantization_default_README.html">DefaultQuantization</a> and precise <a href="https://docs.openvino.ai/latest/pot_compression_algorithms_quantization_accuracy_aware_README.html">AccuracyAwareQuantization</a>.
- Uses the  Tree-structured  Parzen Estimator for global optimization of post-training quantization parameters
- Supports both symmetrical and asymmetrical quantization
- Supports compression for multiple hardware platforms (CPU, GPU)
- Quantizes all channels at the convolutional layer and full connection layer
- Supports multiple applications: computer vision, recommendation system
- Provides customized optimization methods through provided API


Please refer to the following websites for instructions of operations and use:

<a href="https://docs.openvino.ai/latest/pot_README.html">Introduction to the Post-Training Optimization Toolkit</a>

<a href="https://docs.openvino.ai/latest/pot_docs_LowPrecisionOptimizationGuide.html">Low Precision Optimization Guide</a>

<a href="https://docs.openvino.ai/latest/pot_docs_BestPractices.html">Post-training Optimization Toolkit Best Practices</a>

<a href="https://docs.openvino.ai/latest/pot_docs_FrequentlyAskedQuestions.html">Post-training Optimization Toolkit Frequently Asked Questions</a>

<a href="https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html">INT8 quantization and optimization using DL Workbench&rsquo;s web interface</a>

## Using Intel&reg; DAAL for Accelerated Machine Learning

<a id="_Hlk40274671"></a>Intel<sup>&reg;</sup> Data Analytics Acceleration Library (Intel&reg; DAAL)

As a branch of artificial intelligence, machine learning is currently attracting a huge amount of attention. Machine learning-based analytics is also getting increasingly popular. The reason is that, when compared to other analytics, machine learning can help IT staff, data scientists, and various business teams and their organizations to quickly unleash the strengths of AI. Furthermore, machine learning offers many new commercial and open-source solutions, providing a vast ecosystem for developers. In addition, developers can choose from a variety of open-source machine learning libraries such as  Scikit-learn,  Cloudera* and  Spark* MLlib.

### <a id="_Toc67652318"></a><a id="_Toc68526437"></a>Intel&reg; Distribution for Python*

Intel&reg; Distribution for Python* is a Python development toolkit for artificial intelligence software developers. It can be used to accelerate computational speed of Python on the Intel&reg; Xeon&reg; Scalable Processor platform. It is available at  Anaconda*, and it can also be installed and used with  Conda*, PIP*, APT GET, YUM, Docker*, among others. Reference and download site: <a href="/content/www/us/en/developer/tools/oneapi/distribution-for-python.html">https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html</a>

Intel&reg; Distribution for Python* features:


- Out-of-the-box: no or little change to source code required to achieve faster Python application performance.
- The Integrated Intel&reg; performance libraries: Intel&reg; Math Kernel Library (MKL) and Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL), for example, can be used to accelerate NumPy, SciPy, and scikit-learn*
- Latest vector and multithread instructions: Numba* and Cython can be combined to improve concurrency and vectorization efficiency.


### Intel&reg; DAAL

Intel&reg; Data Analytics Acceleration Library (DAAL) is designed for data scientists to accelerate data analytics and prediction efficiency. In particular, it can take full advantage of vectorization and multithreading for applications with huge amount of data, as well as utilize other technologies to increase the overall performance of machine learning on the Intel&reg; Xeon&reg; Scalable Processor platform.

Intel&reg; DAAL is a complete end-to-end software solution designed to help data scientists and analysts quickly build everything from data pre-processing, to data feature engineering, data modeling and deployment. It provides various data analytics needed to develop machine learning and analytics as well as high-performance building blocks required by algorithms. It currently supports linear regression, logic regression, LASSO, AdaBoost, Bayesian classifiers, support vector machines, k-nearest neighbors, k-means clustering, DBSCAN clustering, various types of decision trees, random forest, gradient boosting, and other classic machine learning algorithms. These algorithms are highly optimized to achieve high performance on Intel&reg; processors. For example, a leading big data analytics technology and service provider has used these resources to improve the performance of data mining algorithms by several times.

<img alt="Intel DAAL" height="326" src="/content/dam/develop/external/us/en/images/dl-daal.jpg" width="921"/>

To make it easier for developers to use Intel&reg; DAAL in machine learning applications in Intel-based environments, Intel has open-sourced the entire project (<a href="https://github.com/intel/daal">https://github.com/intel/daal</a>), and provides full-memory, streaming and distributed algorithm support for different big data scenarios. For example, DAAL Kmeans can be combined with Spark to perform multi-node clustering on a Spark cluster. In addition, DAAL provides interfaces for C++, Java*, and Python.

 DAAL4py

In order to provide better support for Scikitlearn, which is the most widely used with Python, Intel&reg; DAAL provides a very simple Python interface, DAAL4py (please see the open source website for more details: <a href="https://github.com/IntelPython/daal4py">https://github.com/IntelPython/daal4py</a>). It can be used seamlessly with Scikitlearn and provides acceleration for machine learning algorithms at the underlying layer.

Developers do not need to modify the Scikitlearn source code to benefit from the advantages of automatic vectorization and multithreading. DAAL4py currently supports the following algorithms in Scikitlearn:


- Sklearn linear regression, Sklearn ridge regression and logic regression
- PCA
- KMeans
- pairwise_distance
- SVC (SVM classification)


### Installing Intel&reg; Distribution for Python &amp; Intel&reg; DAAL

<a href="/content/www/us/en/developer/tools/oneapi/distribution-for-python.html">Download and install Intel&reg; Distribution for Python</a>* (Intel&reg; DAAL already included)

<a href="/content/www/us/en/developer/articles/guide/intel-daal-2020-install-guide.html">Installing Intel&reg; DAAL separately</a>

<a href="/content/www/us/en/develop/documentation/dal-developer-guide/top.html">Intel&reg; DAAL Developer Guide</a>

### Using Intel&reg; DAAL

There are two ways to use Intel&reg; DAAL to accelerate scikit-learn:

Method 1: Using the command line


```
 # python -m daal4py &lt;your-scikit-learn-script&gt; 
```

Method 2: Adding it to source code


``` 
import daal4py.sklearn

daal4py.sklearn.patch_sklearn('kmeans') 
```

## References

[1] Intel&reg; AVX-512 info: <a href="https://colfaxresearch.com/skl-avx512/">https://colfaxresearch.com/skl-avx512/</a>

[2] Intel&reg; Optimized AI Frameworks: <a href="/content/www/us/en/developer/tools/frameworks/overview.html">https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html</a>

[3] Intel&reg; Distribution of OpenVINO&trade; toolkit: <a href="https://docs.openvino.ai">https://docs.openvino.ai</a>

[4] Intel&reg; Analytics Zoo: <a href="https://github.com/intel-analytics/analytics-zoo">https://github.com/intel-analytics/analytics-zoo</a>

[5] Hands-on IDP and Intel&reg; DAAL

[6] IDP benchmarks

[7] Intel&reg; DL Boost

[8] Intel&reg; DL Boost: <a href="https://www.intel.com/content/dam/www/public/us/en/documents/product-overviews/dl-boost-product-overview.pdf">https://www.intel.com/content/dam/www/public/us/en/documents/product-overviews/dl-boost-product-overview.pdf</a>

<a href="/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html">https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html</a>

[9] Open source of Faiss project: <a href="https://github.com/facebookresearch/faiss">https://github.com/facebookresearch/faiss</a>

[10] Open source of HNSWLib project: <a href="https://github.com/nmslib/hnswlib">https://github.com/nmslib/hnswlib</a>
