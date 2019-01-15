#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
##BSUB -q gpuk80
### -- set the job Name --
#BSUB -J ChromaAI_wide
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 7:00
# request 16GB of memory
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
#BSUB -o gpu-wide.out
###BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load course.02456/20181201
module load cudnn

/appl/cuda/9.1/samples/bin/x86_64/linux/release/deviceQuery

# commands to execute
python3 chroma.py train --model=unet-v1 --epochs=200 --images-path=all_data/wide --save=model.widev3 --save-frequency=5 --device=cuda --num-workers=4
