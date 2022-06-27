# test data for Intel in order of appearance on index page
expectedURLS = [["https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html","Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors"],
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
["https://www.intel.com/content/www/us/en/developer/articles/guide/xeon-performance-tuning-and-solution-guides.html#gs.nz1hum","Intel® Xeon® Performance Tuning and Solutions"],
["https://www.intel.com/content/www/us/en/developer/articles/tuning-guide-for-genomics-analytics.html","Tuning Guide for Intel® Select Solutions for Genomics Analytics"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/lammps-tuning-guide.html","LAMMPS Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"]]

foundURLS = ['https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html', 'https://www.intel.cn/content/www/cn/zh/developer/articles/technical/deep-learning-with-avx512-and-dl-boost.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/dl-boost-with-openvino-tuning-guide-on-xeon-system.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/spark-tuning-guide-on-xeon-based-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/mongodb-tuning-guide-on-xeon-based-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/open-source-database-tuning-guide-on-xeon-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/redis-tuning-guide-on-xeon-based-systems.html', 'https://www.intel.com/content/www/cn/zh/developer/articles/guide/redis-tuning-guide-on-xeon-based-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/rocksdb-tuning-guide-on-xeon-based-system.html', 'https://www.intel.com/content/www/cn/zh/developer/articles/guide/rocksdb-tuning-guide-on-xeon-based-system.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/rocksdb-benchmarking-with-xeon-based-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/sql-server-tuning-guide-for-otp-using-xeon.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/hpc-cluster-tuning-on-3rd-generation-xeon.html', 'https://www.intel.com/content/www/us/en/developer/articles/tuning-guide-for-genomics-analytics.html', 'https://www.intel.com/content/www/us/en/developer/articles/reference-implementation/recipe-build-and-run-namd-on-intel-xeon-processors-on-single-node.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/relion-3-1-tuning-guide-on-xeon-based-platforms.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/scalable-video-hevc-transcode-tuning-guide.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/data-compression-tuning-guide-on-xeon-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/wordpress-tuning-guide-on-xeon-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/java-tuning-guide-for-3rd-gen-xeon-based-platforms.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/kvm-tuning-guide-on-xeon-based-systems.html', 'https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-intel-optimization-for-mxnet.html', 'https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html', 'https://software.intel.com/content/www/us/en/develop/download/intel-64-and-ia-32-architectures-optimization-reference-manual.html', 'https://www.intel.com/content/www/us/en/developer/articles/guide/processor-specific-performance-analysis-papers.html', 'https://www.intel.com/content/www/us/en/developer/articles/technical/automated-sku-selection-for-intel-xeon-processors.html', 'https://www.intel.com/content/www/us/en/developer/articles/technical/third-generation-xeon-scalable-family-overview.html', 'https://software.intel.com/content/www/us/en/develop/download/openhpc-installation-guide-for-centos.html']

print("EXPECTED: ", len(expectedURLS))
# expectedURLS is a list of lists
# create a list of urls
expecteds = []
for x in expectedURLS:
    #print(x[0])
    expecteds.append(x[0])

print("FOUND: ", len(foundURLS))
#for x in foundURLS:
#    print(x)

# cast list as tuple then cast as a set
found = tuple(foundURLS)
f = set(found)
expect =tuple(expecteds) 
e = set(expect)
moreFound = list(f-e)
moreExpected = list(e-f)

print("\nMore URLs were found than expected: ", len(moreFound))
print(" ")
for m in moreFound:
    print(m)
print(" ")
    
print("Not all URLs were found.  Missing URLs: ", len(moreExpected))
print(" ")
for m in moreExpected:
    print(m)


#otherURLS = ['https://www.intel.com/content/www/us/en/developer/articles/guide/https://www.intel.com/content/www/us/en/developer/articles/guide/xeon-performance-tuning-and-solution-guides.html#AI Headline', 'https://cdrdv2.intel.com/v1/dl/getContent/684945', 'https://software.intel.com/content/dam/develop/external/cn/zh/documents/Deep-Learning-with-Intel-AVX512-and-Intel-Deep-Learning-Boost-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/openvino-toolkit-tuning-guide-on-xeon.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/spark-tuning-guide-on-xeon.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/MongoDB-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://cdrdv2.intel.com/v1/dl/getContent/684953', 'https://cdrdv2.intel.com/v1/dl/getContent/685354', 'https://software.intel.com/content/dam/develop/external/cn/zh/documents/Redis-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors-Intel-Optane-Persistent-Memory.pdf', 'https://cdrdv2.intel.com/v1/dl/getContent/685340', 'https://software.intel.com/content/dam/develop/external/us/en/documents/rocksdb-benchmark-tuning-guide-on-xeon.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/sql-tuning-guide-on-xeon.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/HPC-Cluster-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://www.intel.com/content/dam/develop/external/us/en/documents/Tuning%20Guide%20for%20Genomics%20Analitics.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/relion-tuning-guide-on-xeon-v2.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/SVT-HEVC-Encoder-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/data-compression-tuning-guide-on-xeon.pdf', 'https://www.intel.com/content/www/us/en/developer/articles/guide/nginx-https-with-crypto-ni-tuning-guide.html', 'https://software.intel.com/content/dam/develop/external/us/en/documents/Nginx-HTTPs-with-Crypto-NI-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://www.intel.com/content/www/cn/zh/developer/articles/guide/nginx-https-with-crypto-ni-tuning-guide.html', 'https://software.intel.com/content/dam/develop/external/cn/zh/documents/Nginx-HTTPs-with-Crypto-NI-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://www.intel.com/content/www/us/en/developer/articles/guide/nginx-https-with-qat-tuning-guide.html', 'https://software.intel.com/content/dam/develop/external/us/en/documents/Nginx-HTTPs-with-QAT-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://www.intel.com/content/www/cn/zh/developer/articles/guide/nginx-https-with-qat-tuning-guide.html', 'https://software.intel.com/content/dam/develop/external/cn/zh/documents/Nginx-HTTPs-with-QAT-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/WordPress-Tuning-Guide-on-3rd-Generation-Intel-Xeon-Scalable-Processors.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/java-tuning-guide-on-xeon.pdf', 'https://software.intel.com/content/dam/develop/external/us/en/documents/kvm-tuning-guide-on-xeon-systems.pdf', 'https://community.intel.com/t5/Software-Tuning-Performance/bd-p/software-tuning-perf-optimization', 'http://www.intel.com/PerformanceIndex']
#for u in otherURLS:
#    print(u)


