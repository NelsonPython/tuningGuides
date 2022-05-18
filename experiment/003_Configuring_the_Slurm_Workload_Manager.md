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

