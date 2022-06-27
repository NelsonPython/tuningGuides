##################
# Check for broken links
# - read the stream and ignore everything before:  <div class="editorialBody">
#     and ignore everything after:  <"/main">
# - follow each URL and record the response code
# - when verbose is 0 report response codes other than 200 or 403
#       when verbose is 1 report all response codes
# - match HTML page name with author
#       report the: author's email, html file name, sentence, and error
#
# TODO:
# 1. group the errors under author email and html page to avoid redundancy
# 2. Parse the tuning guide title from this tag:
# <h1 class="editorialTitle"> Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors </h1>
#
# REFERENCES:
# 1. https://stackoverflow.com/questions/24153519/how-to-read-html-from-a-url-in-python-3
#       sample:  url = "http://www.memidex.com/"
# 2. https://stackoverflow.com/questions/1726402/in-python-how-do-i-use-urllib-to-see-if-a-website-is-404-or-200
#########################

import os
import urllib.request

def followURL(url,verbose):
    msg = ''
    reform = ''
    broken = 0
    
    if '"/>' in url:
        reform = "BAD URL: "+url+","
        u = url.split('"')
        url = u[0]
        reform += " REFORMATTED: "+url+","

    # removing dots at the end
    if url[-1:len(url)] == '.':
        url = url[:-1]

    # remove clutter after .html or .pdf
    if url[:-4] != 'html' or url[:-3] != 'pdf':
        url_list = url.split(".")
        prefx = url_list[-1].split("%")
        url = ''

        for x in range(len(url_list)-1):
            url += url_list[x]+"."
        url+= prefx[0]
            
    # pick one user_agent and add it to the header.  possible user agents:
    # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent':user_agent,}

    stopFlag = 0
    try:
        request = urllib.request.Request(url,None,headers)
    except Exception as e:
        msg = reform+" MALFORMED URL: "+url+", "+" ERROR: ",str(e)+","
        broken = 1
        return broken, msg

    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        # print specific HTTP errors
        if str(e.code) == '403':
            if verbose == 1:
                msg = reform+str(e.code)+" "+url+","
                return broken, msg
            else:
                # return 999 because verbose != 1 and valid links should not print
                return broken, '999'
        else:                
            msg = reform+str(e.code)+" "+url+","
            broken = 1
            return broken, msg
    except urllib.error.URLError as e:
        # print other errors
        msg = reform+str(e.reason)+" "+url+","
        broken = 1
        return broken, msg
    else:
        # if no errors then HTTP 200 is expected
        if verbose == 1:
            res = response.getcode()
            response.close()
            msg = reform+str(res)+" "+url+","
            return broken, msg
        else:
            # return 999 because verbose != 1 and valid links should not print
            return broken, '999'


def printIt(line, msg, filename, logFile):
    # lookup author
    author_email = authors[filename]
    print("AUTHOR: ", author_email, file=logFile)
    print('HTMLFILE: ',filename, file=logFile)    
    print("SENTENCE: ",line, file=logFile)
    print("MESSAGE: ", msg, file=logFile)
    print(" ", file=logFile)

    #print("AUTHOR: ", author_email)
    #print('HTMLFILE: ',parsed)    
    #print("SENTENCE: ",line)
    #print("MESSAGE: ", msg)

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
["https://www.intel.com/content/www/us/en/developer/articles/guide/xeon-performance-tuning-and-solution-guides.html#gs.nz1hum","Intel® Xeon® Performance Tuning and Solutions"],
["https://www.intel.com/content/www/us/en/developer/articles/tuning-guide-for-genomics-analytics.html","Tuning Guide for Intel® Select Solutions for Genomics Analytics"],
["https://www.intel.com/content/www/us/en/developer/articles/guide/lammps-tuning-guide.html","LAMMPS Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"]]


#urls = [["https://www.intel.com/content/www/us/en/developer/articles/guide/lammps-tuning-guide.html","LAMMPS Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform"]]


# 0-do not save; 1-save the file
saveFile = 0
# 0-error; 1-all
verbose = 0
# counts the total number of links followed
K = 0
# sums the number of broken links
b = 0
# receives the broken link counter update from the function
brokenK = 0
# HTTP status message
msg = ''
# skip the header and footer and only read the content
readFlag = 0

# load a dictionary with author's email addresses and the page they are responsible for
# use this to lookup the email address when a broken link must be reported
import csv
with open('/home/nelson/TuningGuides/0-resources/pages2authors.csv', mode='r') as infile:
#with open('C:\\Users\\SDene\\AppData\\Local\\Programs\\Python\\Python37\\0-intel_tuning_guides\\authors\\pages2authors.csv', mode='r') as infile:
    reader = csv.reader(infile)
    authors = {rows[0]:rows[1] for rows in reader}

logFile = open("/home/nelson/TuningGuides/0-log/tuningGuideLink.log","w")
#logFile = open("C:\\Users\\SDene\\AppData\\Local\\Programs\\Python\\Python37\\0-intel_tuning_guides\\log\\tuningGuideLink.log","w")

for url in urls:
    httpURL = url[0]

    # pick one user_agent and add it to the header.  possible user agents:
    # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent':user_agent,} 
    request = urllib.request.Request(httpURL,None,headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    text = data.decode("utf8")
    response.close()

    parseURL = httpURL.split("/")
    # one URL has an internal link denoted with #
    # parsed[0] will be the name of the file such as: lammps-tuning-guide.html
    parsed = parseURL[-1].split("#")

    if saveFile == 1:
        fileURL = "0-intel_tuning_guides/"+parsed[0]
        print(httpURL," COPIED TO: ",fileURL,"\n")
        fo = open(fileURL, "w", encoding='utf8')
        print(text, file=fo)
        fo.close()
   
    lines = text.split('\n')
    for line in lines:
        if '<div class="editorialBody">' in line:
            # start looking for URLs
            readFlag = 1

        if '</main>' in line:
            # stop looking for URLs
            readFlag = 0

        if readFlag == 1:
            if '<a href' in line:
                K+=1
                a = line.split('<a href="')
                a = a[1].split('">')
                url = a[0]

                if url.startswith('/content/www/') or url.startswith('/content/dam/'):
                    url = "https://www.intel.com" + url

                if url.startswith('/file'):
                    url = "https://www.intel.com" + url

                if '"' in url:
                    # title=", target="
                    u = url.split('"')
                    url = u[0]

                if url[0] == '#':
                    url = "https://www.intel.com/content/www/us/en/developer/articles/guide/" + httpURL + url

                brokenK, msg = followURL(url,verbose)
                b += int(brokenK)
                if msg != '999':
                    printIt(line, msg, parsed[0], logFile)

            if 'src="' in line:
                p = line.split('src="')
                p = p[1].split(' width')
                url = "https://www.intel.com" + p[0][:-1]
                if "etc.clientlibs" not in url:
                    K+=1
                    brokenK, msg = followURL(url,verbose)
                    b += int(brokenK)
                if msg != '999':
                    printIt(line, msg, parsed[0], logFile)

print("Total links followed: ", K, file=logFile)
print("Total broken links: ", b, file=logFile)
print("======================================================", file=logFile)
logFile.close()
