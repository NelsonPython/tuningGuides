# https://stackoverflow.com/questions/24153519/how-to-read-html-from-a-url-in-python-3
# sample:  url = "http://www.memidex.com/"
#
# ToDo:  parse out tuning guide title the <h1 class="editorialTitle">:
#
#<h1 class="editorialTitle"> Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors </h1>
#
#########################

import urllib.request
'''
urls = [["https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html","Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html","借助基于第 3 代英特尔® 至强® 可扩展处理器的英特尔® AVX-512 和英特尔® 深度学习加速调优指南进行深度学习"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/dl-boost-with-openvino-tuning-guide-on-xeon-system.html","Intel® Distribution of OpenVINO™ Toolkit Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/spark-tuning-guide-on-xeon-based-systems.html","Spark Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/mongodb-tuning-guide-on-xeon-based-systems.html","MongoDB* Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/open-source-database-tuning-guide-on-xeon-systems.html","Open-Source Database Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/redis-tuning-guide-on-xeon-based-systems.html","Redis* Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors & Intel® Optane™ Persistent Memory"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/redis-tuning-guide-on-xeon-based-systems.html","Redis* 优化指南 - 基于第 3 代英特尔® 至强® 可扩展处理器及英特尔® 傲腾™ 持久内存"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/rocksdb-tuning-guide-on-xeon-based-system.html","RocksDB* Tuning Guide on Intel® Xeon® Scalable Processors"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/rocksdb-tuning-guide-on-xeon-based-system.html","RocksDB* 优化指南 - 基于第 3 代英特尔® 至强® 可扩展处理器"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/rocksdb-benchmarking-with-xeon-based-systems.html","RocksDB* Benchmarking Tuning Guide with 3rd Generation Intel® Xeon® Scalable Processor Platforms"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/sql-server-tuning-guide-for-otp-using-xeon.html","Microsoft* SQL Server* Tuning Guide for Online Transaction Processing workload on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/hpc-cluster-tuning-on-3rd-generation-xeon.html","HPC Cluster Tuning on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.com/content/www/us/en/developer/articles/reference-implementation/recipe-build-and-run-namd-on-intel-xeon-processors-on-single-node.html","Recipe: Build and Run NAMD on Intel® Xeon® Processors"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/relion-3-1-tuning-guide-on-xeon-based-platforms.html","RELION* 3.1 Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/scalable-video-hevc-transcode-tuning-guide.html","SVT-HEVC Encoder Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/data-compression-tuning-guide-on-xeon-systems.html","Data Compression with Intel® ISA-L/ Intel® IPP/ Intel® QAT Tuning Guide"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/nginx-https-with-crypto-ni-tuning-guide.html","Nginx* HTTPs with Crypto-NI Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/nginx-https-with-crypto-ni-tuning-guide.html","Nginx* HTTPS 优化指南 - 基于第 3 代英特尔® 至强® 可扩展处理器及 Crypto-NI 技术"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/nginx-https-with-qat-tuning-guide.html","Nginx* HTTPs with Intel® QAT Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/nginx-https-with-qat-tuning-guide.html","Nginx* HTTPS 优化指南 - 基于第三代英特尔® 至强® 可扩展处理器及英特尔® QuickAssist 技术"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/wordpress-tuning-guide-on-xeon-systems.html","WordPress* Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/java-tuning-guide-for-3rd-gen-xeon-based-platforms.html","Java* Tuning Guide for 3rd Generation Intel® Xeon® Processor Platforms"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/kvm-tuning-guide-on-xeon-based-systems.html","KVM/Qemu Virtualization Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html","Maximize TensorFlow* Performance on CPU: Considerations and Recommendations for Inference Workloads"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html","Guide to TensorFlow* Runtime optimizations for CPU"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-intel-optimization-for-mxnet.html","Getting Started with Intel® Optimization for MXNet*"],
["https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html","Code Sample: Intel® Deep Learning Boost New Deep Learning Instruction bfloat16 - Intrinsic Functions"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/processor-specific-performance-analysis-papers.html","Tuning Guides and Performance Analysis Papers"],
["https://www.intel.com/content/www/us/en/developer/articles/technical/automated-sku-selection-for-intel-xeon-processors.html","Automated SKU Selection for Intel® Xeon® Processors through Machine Learning"],
["https://www.intel.com/content/www/us/en/developer/articles/technical/third-generation-xeon-scalable-family-overview.html","Third Generation Intel® Xeon® Processor Scalable Family On Two Socket Platform Technical Overview"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/xeon-performance-tuning-and-solution-guides.html#gs.nz1hum","Intel® Xeon® Performance Tuning and Solutions"]]
'''

urls = [["https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html","Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
["https://www.intel.cn/content/www/cn/zh/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html","借助基于第 3 代英特尔® 至强® 可扩展处理器的英特尔® AVX-512 和英特尔® 深度学习加速调优指南进行深度学习"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/dl-boost-with-openvino-tuning-guide-on-xeon-system.html","Intel® Distribution of OpenVINO™ Toolkit Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"],
["https://www.intel.com/content/www/us/en/developer/articles/tuning-guide-for-genomics-analytics.html","Tuning Guide for Intel® Select Solutions for Genomics Analytics"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/lammps-tuning-guide.html","LAMMPS Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"]]


for url in urls:
    httpURL = url[0]

    parseURL = httpURL.split("/")
    # one URL has an internal link denoted with #
    parsed = parseURL[-1].split("#")
    fileURL = "0-intel_tuning_guides/"+parsed[0]

    print(httpURL," COPIED TO: ",fileURL,"\n")

    # pick one user_agent and add it to the header.  possible user agents:
    # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent':user_agent,} 
    request = urllib.request.Request(httpURL,None,headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    mystr = data.decode("utf8")
    response.close()

    fo = open(fileURL, "w", encoding='utf8')
    print(mystr, file=fo)
    fo.close()

