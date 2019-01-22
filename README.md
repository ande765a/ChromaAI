# ChromaAI
A deep neural network for colorizing greyscale images. DTU Project.

The models used in our paper is (from `model.py`) `ColorizerV3` as the reg. CNN and `ColorizerUNetV1` as U-Net. We kept the other models

## Usage in a terminal
To use our program in a terminal run the command, followed by arguments
```sh
python mode [-h] [--output OUTPUT] [--images-path IMAGES_PATH]
            [--model MODEL] [--load LOAD] [--save SAVE]
            [--log-output LOG_OUTPUT] [--loss LOSS] [--no-l NO_L]
            [--save-best SAVE_BEST] [--save-frequency SAVE_FREQUENCY]
            [--device DEVICE] [--learning-rate LEARNING_RATE]
            [--num-workers NUM_WORKERS] [--batch-size BATCH_SIZE]
            [--epochs EPOCHS] [--shuffle SHUFFLE]
```

Argument                         | Usage
-------------------------------- | ----------------------------------------------------------------------------
mode                             | Eval for evaluation or train for training mode
-h                               | Shows the avaliable arguments
--output OUTPUT                  | Output path when evaluating
--images-path IMAGES_PATH        | Path for the image dataset, should be a folder with 2 sub-folders, validation and training
--model MODEL                    | Specify which model to load [v1 \| v2 \| v3 \| v3 \| unet-v1 \| unet-v2 \| unet-v3]
--load LOAD                      | Path for loading trained parameters
--save SAVE                      | Path for saving trained parameters
--log-output LOG_OUTPUT          | Path for saving a .csv log over the training and validation loss
--loss LOSS                      | Specify the loss function to use either gan or mse, defaults to mse: Mean Squared Error
--no-l NO_L                      | Either True or False, when evaluating specify if you only want A&B channels
--save-best SAVE_BEST            | Path for saving the best trained parameteres
--save-frequency SAVE_FREQUENCY  | How often the program should save the parameters
--device DEVICE                  | Either cuda:0 or cpu, defaults to cpu
--learning-rate LEARNING_RATE    | Specify a learning to use when training
--num-workers NUM_WORKERS        | The number of workers to use (rule of thumb: use number of cpu-cores)
--batch-size BATCH_SIZE          | Specify the batch size to use when trainig, defaults to 32
--epochs EPOCHS                  | Number of epochs to train
--shuffle SHUFFLE                | Either True or False, should the program shuffle the dataloading


If there are multiple cuda devices available the program will automaticaly make use of them with the `torch.nn.DataParallel` wrapper.

## Usage with DTU HPC GPU node<sup>[1](#footnote)</sup>
To submit at job to DTU's HPC GPU `gpuv100` node, login to one of the following 
nodes: `login2.hpc.dtu.dk`, `login3.hpc.dtu.dk` or `login3.gbar.dtu.dk` via an 
ssh connection as follows.
```sh
ssh user@login2.hpc.dtu.dk
```
where `user` is either your credentials (for employees) or studentid.
From here use the command `linuxsh` to go from the login-node to a 
general-purpose-node. Here you can execute one of the jobscripts with the following command
```bash
bsub < jobscript_train_wide.sh
```
This will send the job to the node configured in the job script.

The job script is configured to save the outputs of the program it executes in
a file named `gpu-id.out`. `id` is the job id on the gpu-node.

## Authors
We are 3 DTU Students from the course Introduction to Intelligent Systems

s174300 Anne Agathe Pedersen

s183903 Niels Kjær Ersbøll 

s183926 Anders Bredgaard Thuesen

## Resources
<a name="footnote">[1]</a>: DTU's guide for [Using GPUs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)
