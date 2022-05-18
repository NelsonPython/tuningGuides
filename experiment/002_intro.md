## Introducing Intel® Select Solutions for Genomics Analytics

This guide focuses on software configuration recommendations for users who are already familiar with the Intel® Select Solutions for Genomics Analytics.  The following guide contains the hardware configuration for the HPC clusters that are used to run this software:  HPC Cluster Tuning on 3rd Generation Intel® Xeon® Scalable Processors.    However, please carefully consider all of these settings based on your specific scenarios.  Intel® Select Solutions for Genomics Analytics can be deployed in multiple ways and this is a reference to one use-case. 

Genomics analysis is accomplished using a suite of software tools including the following:
- Genomic Analysis Toolkit (GATK) Version 4.1.9.0 is used for variant discovery. GATK is the industry standard for identifying SNPs and indels in germline DNA and RNAseq data.  It includes utilities to perform tasks such as processing and quality control of high-throughput sequencing data.
- Cromwell Version 52 is a Workflow Management System for scientific workflows.  These workflows are described in the Workflow Description Language (WDL).  Cromwell tracks workflows and stores the output of tasks in a MariaDB database.
- Burrows-Wheeler Aligner (BWA) Version 0.7.17 is used for mapping low-divergent sequences against a large reference genome, such as the human genome.
- Picard Version 2.23.8 is set of tools used to manipulate high-throughput sequencing (HTS) data stored in various file formats such as: SAM, BAM, CRAM, or VCF.
- VerifyBAMID2 works with Samtools Version 1.9 to verify whether sample genomic data matches previously known genotypes.  It can also detect sample contamination.
- Samtools Version 1.9 are used to interact with high-throughput sequencing data including:  reading, writing, editing, indexing, or viewing data stored in the SAM, BAM, or CRAM format.  It is also used for reading or writing BCF2, VCF, or gVCF files and for calling, filtering, or summarizing SNP and short indel sequence variants.
- 20K Throughput Run is a simple and quick benchmark test used to ensure that the most important functions of your HPC cluster are configured correctly.
- Optional:  Intel® System Configuration Utility (SYSCFG) Version 14.2 Build 8 command-line utility can be used to save and to restore system BIOS and management firmware settings.

These prerequisites are required:  
- Git is a version control system for tracking changes.
- Either Java 8 or the Java Runtime Environment (JRE) plus the Software Developers Kit (SDK) 1.8 is required to run Cromwell.
- Python Version 2.6 or greater is required for the gatk-launch.  Newer tools and workflows require Python 3.6.2 along with other Python packages.  Use the Conda package manager to manage your environment and dependencies.  
- Slurm Workload Manager provides a means of scheduling jobs and allocating cluster resources to those jobs.  For a detailed description of Slurm, please visit: https://slurm.schedmd.com/documentation.html.
- sbt is required to compile Cromwell.
- MariaDB is used by Cromwell for persistent storage.
- R, Rscript, gsalib, ggplot2, reshape, and gplots are used by GATK to produce plots.  

3rd Generation Intel® Xeon® Scalable processors deliver platforms with built-in AI acceleration that can be optimized for many workloads.  They provide a performance foundation to help speed data’s transformative impact, from a multi-cloud environment to the intelligent edge and back.  Improvements of particular interest to this workload are: 
- Enhanced Performance
- Increased DDR4 Memory Speed & Capacity
- Intel® Advanced Vector Extensions
- Intel® Genomics Kernel Library with AVX-512

### Architecture 

Scientists use tools such as, the Genomic Analysis Toolkit, along with WDL scripts to process DNA and RNA data.  Cromwell is used to manage the workflow.  SLURM is used to schedule jobs and to allocate cluster resources to those jobs.  This diagram shows the software and hardware used in the Intel® Select Solutions for Genomics Analytics.

![Diagram showing tools explained in this paragraph](images/Genomics-Architecture-diagram.png)

 
## Tuning the Intel® Select Solutions for Genomics Analytics

Software configuration tuning is essential because the operating system and the genomics analysis software were designed for general-purpose applications.  The default configurations have not been tuned for the best performance.  The following sections provide step-by-step guidance for tuning the Intel® Select Solutions for Genomics Analytics.

### Tuning the HPC Cluster

Tuning recommendations for hardware are addressed in this guide:  HPC Cluster Tuning on 3rd Generation Intel® Xeon® Scalable Processors at:  https://www.intel.com/content/www/us/en/developer/articles/guide/hpc-cluster-tuning-on-3rd-generation-xeon.html  
