# ChromaAI
A deep neural network for colorizing greyscale images. DTU Project.

## Usage with DTU HPC GPU node[^1]
To submit at job to DTU's HPC GPU `gpuv100` node, login to one of the following 
nodes: `login2.hpc.dtu.dk`, `login3.hpc.dtu.dk` or `login3.gbar.dtu.dk` via an 
ssh connection as follows.
```
ssh user@login2.hpc.dtu.dk
```
where `user` is either your credentials (for employees) or studentid.
From here use the command `linuxsh` to go from the login-node to a 
general-purpose-node. Here you can execute the `jobscript.sh` with the following
command
```
bsub < jobscript.sh
```
This will send the job to the node configured in the job script.

The job script is configured to save the outputs of the program it executes in
a file named `gpu-id.out`. `id` is the job id on the gpu-node.

[^1]: DTU's guide for [Using GPUs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)