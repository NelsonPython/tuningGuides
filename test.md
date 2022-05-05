In this post, we'll use VMWare ESXi 7.0u3 to create a Guest VM running Ubuntu 21.10 with two Virtual Persistent Memory (vPMem) devices, then show how we can map the vPMem device in the host (ESXi) to "nmem" devices in the guest VM as shown by the `ndctl` utility.

