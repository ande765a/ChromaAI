#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
##BSUB -q gpuk80
### -- set the job Name --
#BSUB -J ChromaAI_narrow
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 8GB of memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u nker@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-narrow.out
###BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load course.02456/20181201
module load cudnn

/appl/cuda/9.1/samples/bin/x86_64/linux/release/deviceQuery

# commands to execute
python3 chroma.py train --epochs=8000 --images-path=all_data/narrow --save=model.narrow --save-frequency=5 --load=model.narrow --device=cuda --num-workers=4
