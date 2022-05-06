# Intel® Tuning Guides Markdown Template

Answer questions in (parenthesis)

## Introduction

This guide is for users who are already familiar with (the workload).  It provides recommendations for configuring hardware and software that will provide the best performance in most situations. However, please note that we rely on the users to carefully consider these settings for their specific scenarios, since (the workload) can be deployed in multiple ways. 
	
(Describe the workload including version numbers used for testing and list links to get more information.)

(Describe other software used in the guide including the version numbers and links to more information.)
	
(Describe pre-requisites)
	
3rd Generation Intel® Xeon® Scalable processors deliver industry-leading, workload-optimized platforms with built-in AI acceleration.  They provide a seamless performance foundation to help speed the transformative impact of data from the intelligent edge to the multi-cloud. Improvements of particular interest to this workload are: (select those that apply to running this workload)
- Enhanced Performance
- Enhanced Intel® Deep Learning Boost with VNNI
- More Intel® Ultra Path Interconnect
- Increased DDR4 Memory Speed & Capacity
- Intel® Advanced Vector Extensions
- Intel® Security Essentials and Intel® Security Libraries for Data Center 
- Intel® Speed Select Technology
- Support for Intel® Optane™ Persistent Memory 200 series

Tested hardware and software for this tuning guide include:

### Server Configuration

#### Hardware

The configuration described in this article is based on 3rd Generation Intel® Xeon® processor hardware. The server platform, memory, hard drives, and network interface cards can be determined according to your usage requirements.

| Hardware | Model |
|----------------------------------|------------------------------------|
| Server Platform Name/Brand/Model | Intel® Server System M50CYP1UR212 |
| CPU | Intel® Xeon® PLATINUM 8360Y CPU @ 2.20GHz | 
| BIOS | version # | 
| Memory | 16*32 GB DDR4, 3200 MT/s | 
| Storage/Disks | Intel SSD S4610, 960G | 
| NIC (if it applies) | Intel® Ethernet Controller XXV700 25GbE SFP28 | 

#### Software

| Software | Version |
|------------------|-------------|
| Operating System | CentOS* 7.8 | 
| Kernel | 3.10.0-1127.el7.x86_64 | 
| (Workload) | version # | 
| Other SW used | version # | 
| Other SW used | version # | 
| Other SW used | version # | 

			
## Hardware Tuning

(Enter any other information on general system setup and tuning.)

### BIOS Settings

(Use this section to describe BIOS settings that should be changed in order to improve workload performance.)

Begin by resetting your BIOS to default setting, then follow the suggestions below for changes to the default:

| Configuration Item | Recommended Value
|---------------------|------------------|
| Advanced/Power & Performance/CPU P State Control/CPU P State Control/Enhanced Intel SpeedStep® Tech | Disabled |
| Advanced/Power & Performance/CPU Power and Performance Policy | Performance |
| Advanced/Memory Configuration/SNC (Sub-NUMA Clusters) | Enabled |
| Advanced/Memory Configuration/Page Policy | Closed |
| Advanced/UPI Configuration/XPT Prefetch | Enabled |
| Advanced/Processor Configuration/Direct-to-UPI (D2K) | Enabled |

If no specific settings apply, then use this text: (No specific BIOS setting for this this workload – just use the defaults)

If you recommend a BIOS setting, give an explanation of what the setting does and why it matters to the workload.  Here is an example from the HPC Cluster Tuning Guide:

### Description of Settings

#### Sub-NUMA Cluster (SNC) 

SNC is a feature that provides similar localization benefits as Cluster-On-Die (COD), a feature found in previous processor families, without some of COD’s downsides. SNC breaks up the last level cache (LLC) into disjoint clusters based on address range, with each cluster bound to a subset of the memory controllers in the system. SNC improves average latency to the LLC and is a replacement for the COD feature found in previous processor families.

#### Direct-to-UPI (D2K)

D2U is a latency-saving feature for remote read transactions. With D2U enabled, the IMC will send the data directly to the UPI instead of going through the Caching and Home Agent (CHA), reducing latency. Keep enabled, although workloads that are highly NUMA-optimized or that use high levels of memory bandwidth are less likely to be affected by disabling D2U.

#### XPT (eXtended Prediction Table) Prefetch

Extended prediction table (XPT) Prefetch is a new capability that is designed to reduce local memory access latency. XPT Prefetch is an “LLC miss predictor” in each core that will issue a speculative DRAM read request in parallel to an LLC lookup, but only when XPT predicts a “miss” from the LLC lookup.
For more information, refer to the BIOS Setup Utility User Guide for the Intel® Server Board D50TNP and M50CYP Family.

### Memory Configuration/Settings

(Use this section to describe the optimum memory configuration.  Here are some questions to consider:  How many DIMMS per channel?  How many channels are used?  Is PMem appropriate for this workload?  If so how should it be configured?)

Example: At least 1 DIMM per memory channel needs to be populated. Lower cpu-utilization could be an issue if there are no DIMMs in a memory channel because of the contention from database cache.

If no specific settings apply, then use this text: (No specific workload setting)

### Storage/Disk Configuration/Settings

(Are there any specific settings or recommendations for storage media?)

If no specific suggestions apply, then use this text: (No specific workload setting)

### Network Configuration/Setting

(Are there any specific settings or recommendations for the network setup?  Does your workload use multiple systems? Any advice on how many clients? For example, how much CPU and memory will they need and how they should it be setup?). 

Example: In the Redis application scenario, performance is usually restricted more by the bandwidth of the network than the performance of memory and Intel persistent memory. Therefore, when you run Redis across networks, you need an NIC with a highest possible network bandwidth. It is recommended that the value is above 10GB/s

If no specific suggestions apply, then use this text: (No specific workload setting for this topic)

## Software Tuning 

Software configuration tuning is essential. From the operating system to (the workload) configuration settings, they are all designed for general purpose applications and default settings are almost never tuned for best performance.

### Linux Kernel Optimization Settings (Replace Linux with another OS if applicable)

Use this section to describe and list commands to issue for OS optimization.  

### (The Workload) Architecture

Use this section to describe the workload architecture and how it works. Insert pictures or drawings as appropriate.  You may choose to host your screenshots, diagrams, terminal window images in your github repo. Just remember, you are now supporting the live site. Don't move or delete images without updating your article. Also, make sure to use the full github URL and not a relative path.

![This is your Alt Text](https://raw.githubusercontent.com/tracyjohnsonidz/devzone-articles/main/diagram-full-workflow-16x9.webp)
 

### (The Workload) Tuning

Enter settings, configurations, to consider. List any commands to be typed into a console with black text and highlight them with a blue background. Example:

```
sysctl -w kernel.sched_domain.cpu(x).domain0.max_newidle_lb_cost=0
sysctl -w kernel.sched_domain.cpu(x).domain1.max_newidle_lb_cost=0
```

## Related Tools and Information (optional section)

In this section you can list tools that are related to the workload or solution such as performance monitoring/testing tools, or configuration checking tools, or platform utility tools. It is up to you if there are any tools you want to tell users about, listing appropriate code examples and screenshots.

## Best Practices for Testing and Verification (optional section)

In this section you can list any BKMs you have for running the workload. Example below is from the WorkPress Tuning Guide:

Example: Note that the recommendations in the guide for WordPress workload are only a reference, and the tunings here should be carefully adopted by someone who is well-versed with the workload and the system settings.

- Since this is a CPU-bound web front-end workload, when all the requests are appropriately distributed, we expect ~90+% CPU-utilization. Use tools like sar/htop to verify you get the expected CPU utilization.
- Execute at least 7 runs to ensure the standard deviation is within 5%.
- Restart MariaDB service after every run to clear query cache. This is specific to the workload and not a recommendation for the real-world web deployments.

## Conclusion

Use this section as a brief wrap-up for the guide. 

Example: We understand every application is unique. We shared many of our experiences with MySQL and PostgreSQL hoping that some of our learnings could be applied to your specific application. Both Open-Source relational database management systems have been well tested on Intel platforms. With 3rd Generation Intel® Xeon® Scalable processor, Intel takes it even further by optimizing the platform as a whole -- CPU, memory, storage, and networking working together for the best user experience.

## Additional Resources (optional, as needed)

## References (optional, as needed)

## Feedback (required section)

We value your feedback. If you have comments (positive or negative) on this guide or are seeking something that is not part of this guide, please reach out and let us know what you think. 
 
