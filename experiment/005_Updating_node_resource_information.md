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
