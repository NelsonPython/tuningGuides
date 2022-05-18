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
### Configuring the Slurm Workload Manager

The Slurm Workload Manager provides a means of scheduling jobs and allocating cluster resources to those jobs.  When adding the Slurm workload manager to the HPC Cluster, install the server component on the frontend node.

#### Installing the Slurm Server on the frontend node
Via Slurm, the PAM (pluggable authentication module) restricts normal user SSH access to compute nodes.  This may not be ideal in certain circumstances.  Slurm needs a system user for the resource management daemons.  The global Slurm configuration file and the cryptographic key that is required by the munge authentication library must be available on every host in the resource management pool.  The default configuration supplied with the OpenHPC build of Slurm requires the "slurm" user account.  Create a  user group and grant unrestricted SSH access.  Add the slurm user to this user group as follows:

1.	Create a user group that is granted unrestricted SSH access:

```
 groupadd sshallowlist  
```

2.	Add the dedicated Intel® Cluster Checker user to the trust-list (previously called whitelist). This user account should be able to run the Cluster Checker both inside and outside of a resource manager job.

```
 usermod -aG sshallowlist clck   
```

3.	Create the Slurm user account:

```
 useradd --system --shell=/sbin/nologin slurm 
```

4.	Install Slurm server packages:

```
 dnf -y install ohpc-slurm-server 
```

5.	Update the Warewulf files:

```
wwsh file sync 
```

##### Summary of the commands

Here are all the commands for this section:

```
groupadd sshallowlist
usermod -aG sshallowlist clck
useradd --system --shell=/sbin/nologin slurm 
dnf -y install ohpc-slurm-server
wwsh file sync
```
#### Updating node resource information

Update the Slurm configuration file with the node names of the compute nodes, the properties of their processors, and the Slurm partitions or queues that are associated with your HPC Cluster, including:
- The NodeName tag, or tags, must reflect the names of the compute nodes along with a definition of their respective capabilities.  
- The Sockets tag defines the number of sockets in a compute node.  
- The CoresPerSocket tag defines the number of CPU cores per socket.  
- The ThreadsPerCore tag defines the number of threads that can be executed in parallel on a core.  
- The PartitionName tag, or tags, defines the Slurm partitions, or queues, where compute names are assigned.  Users use these tags to access these resources.  The name of the partition will be visible to the user.  
- The Nodes argument defines the compute nodes that belong to a given partition.  
- The Default argument defines which partition is the default partition.  The default partition is the partition that is used when a user doesn't explicitly specify a partition.

Complete the following tasks to create a Slurm configuration file for a cluster based on the Bill of Materials for this Reference Design:
- Use the openHPC template to create a new Slurm configuration file.
- Add the host name of the frontend host as the ControlMachine to the Slurm  configuration file.
- Ensure that the SLURM control daemon will make a node available again when the node registers with a valid configuration after being DOWN
- Update the NodeName definition to reflect the hardware capabilities
- Update the PartitionName definition

##### Step-by-step instructions

##### Step 1:  Create

To create a new SLURM config file, copy the template for the openHPC SLURM config file:

```
 cp /etc/slurm/slurm.conf.ohpc /etc/slurm/slurm.conf      
```

##### Step 2:  Update the configuration file	

Open the /etc/slurm/slurm.conf file and make the following changes:

1. Locate and update the line beginning with "ControlMachine" to:

```
ControlMachine=frontend
```

2. The SLURM configuration that ships with OpenHPC has a default set-up.  The  SLURM control daemon will only make a node available after it goes into the DOWN state if the node was in the DOWN state because it was non-responsive.  You can refer to the documentation on the ReturnToService configuration option in:  https://slurm.schedmd.com/slurm.conf.html  This configuration is reasonable for a large cluster under constant supervision by a system administrator.  For a smaller cluster or a cluster that is  not under constant supervision by a system administrator, a different configuration is preferable.  SLURM should make a compute node available again when a node with a valid configuration registers with the SLURM control daemon.  To enable this type of return to service, locate this line:

```
ReturnToService=1
```

Replace it with:

```
ReturnToService=2
```

3. The SLURM configuration that ships with OpenHPC permits sharing a compute node if the specified resource requirements allow it.  Adjust this so that jobs can be scheduled based on CPU and memory requirements.

Locate the line:
  
```
SelectType=select/cons_tres
```

Replace it with the following.  Take care to drop the “t” from tres and use the setting: select/cons_res:

```
SelectType=select/cons_res
```

Locate the line:

```
SelectTypeParameters=CR_Core
```

Replace it with:

```
SelectTypeParameters=CR_Core_Memory
```

4. Update the node information to reflect the cluster configuration.  Locate the NodeName tag.  For compute nodes based on 3rd Generation Intel® Xeon® Scalable Gold 6348 processors, replace the existing line with the following line:

```
NodeName=c[01-04] Sockets=2 CoresPerSocket=28 ThreadsPerCore=2RealMemory=480000 state=UNKNOWN
```

When configured for use with Cromwell, bwa,all and haplo will be used to provide improved performance.  Locate the PartitionName tag and replace the existing line with the following lines.

```
PartitionName=xeon Nodes=c[01-04] Priority=10000 default=YES MaxTime=24:00:00 State=UP
PartitionName=bwa Nodes=c[01-04] Priority=4000 Default=NO MaxTime=24:00:00 State=UP 
PartitionName=all Nodes=c[01-04] Priority=6000 Default=NO MaxTime=24:00:00 State=UP 
PartitionName=haplo Nodes=c[01-04] Priority=5000 default=NO MaxTime=24:00:00 State=UP
```

5. The openHPC slurm.conf example has a typo that prevents slurmctld and slurmd from starting.  Fix that typo by locating the following line and using a “#” symbol to make it a comment:

*Caution:  A second line starting with JobCompType exists. DO NOT change that line!*

Locate this line:

```
JobCompType=jobcomp/none
```

Add the “#” in front to comment it, so that it reads as follows:

```
#JobCompType=jobcomp/none
```

6.	Save and close the file.

##### Step 3: Import

Import the new configuration files into Warewulf:

```
wwsh -y file import /etc/slurm/slurm.conf 
```

##### Step 4: Update the provision configuration file

Update the /etc/warewulf/defaults/provision.conf file:

1. Open the /etc/warewulf/defaults/provision.conf file and located the line starting with:

```
files = dynamic_hosts, passwd, group ...
```

*Note:  Do not change any part of the existing line.  Append the following to the end of the line.  Remember to include a comma at the beginning of the added text.*

```
, slurm.conf, munge.key
```

2. Save and close the file.  

##### Step 5: Enable controller services

Enable MUNGE and Slurm controller services on the frontend node:

```
 syssystemctl enable slurmctld.service 
```

##### Summary of the commands

Here are all the commands for this section:
```
cp /etc/slurm/slurm.conf.ohpc /etc/slurm/slurm.conf

sed -i "s/^\(NodeName.*\)/#\1/" /etc/slurm/slurm.conf                                                     

echo "NodeName=${compute_prefix}[${first_node}-${last_node}] Sockets=${num_sockets} \                     
CoresPerSocket=${num_cores} ThreadsPerCore=2 State=UNKNOWN" >> /etc/slurm/slurm.conf                      
sed -i "s/ControlMachine=.*/ControlMachine=${frontend_name}/" /etc/slurm/slurm.conf                       
sed -i "s/^\(PartitionName.*\)/#\1/" /etc/slurm/slurm.conf                                                
sed -i "s/^\(ReturnToService.*\)/#\1\nReturnToService=2/" /etc/slurm/slurm.conf                           
sed -i "s/^\(SelectTypeParameters=.*\)/#\1/" /etc/slurm/slurm.conf                                        
sed -i "s/^\(JobCompType=jobcomp\/none\)/#\1/" /etc/slurm/slurm.conf                                      
                                                                                                          
cat >> /etc/slurm/slurm.conf << EOFslurm                                                        
                                                                                                
PartitionName=xeon Nodes=${compute_prefix}[${first_node}-${last_node}] Default=YES MaxTime=24:00:00 State=UP                                                                                                  
PartitionName=cluster Nodes=${compute_prefix}[${first_node}-${last_node}] Default=NO MaxTime=24:00:00 \
State=UP
                                                                                   
EOFslurm 
                                                                                                 
wwsh -y file import /etc/slurm/slurm.conf
sed -i "s/^files\(.*\)/files\1, slurm.conf, munge.key/" /etc/warewulf/defaults/provision.conf
syssystemctl enable slurmctld.service 
```
