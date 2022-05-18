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
