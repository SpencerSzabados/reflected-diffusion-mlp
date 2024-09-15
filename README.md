
# Structure Preserving Diffusion Models
This repository contains the implementation(s) for the model(s) proposed in [Structure Preserving Diffusion Models](https://arxiv.org/abs/2402.19369). 

<p align="center">
  <img src="assets/teaser.png" width="60%"/>
</p>

## Abstract 
Diffusion models have become the leading distribution-learning method in recent years. Herein, we introduce structure-preserving diffusion processes, a family of diffusion processes for learning distributions that possess additional structure, such as group symmetry, by developing theoretical conditions under which the diffusion transition steps preserve said symmetry. While also enabling equivariant data sampling trajectories, we exemplify these results by developing a collection of different symmetry equivariant diffusion models capable of learning distributions that are inherently symmetric. Empirical studies, over both synthetic and real-world datasets, are used to validate the developed models adhere to the proposed theory and are capable of achieving improved performance over existing methods in terms of sample equality. We also show how the proposed models can be used to achieve theoretically guaranteed equivariant image noise reduction without prior knowledge of the image orientation.


# Useage
## Environment setup 
We include a [Docker](https://www.docker.com/) buildfile in '/Group-Diffusion/docker' that builds a sutable environment for running all the code by simply running the following docker command (which is also found in '/Group-Diffusion/docker/run_container.sh'). This docker image should download the latest version of the diffusion model code from this repository.

```
$ docker build -t group-diffusion:latest /Group-Diffusion/docker/Dockerfile 
```
You may also use [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), for which you can build the environment using the 'steup.py' file; however, package conflicts are more likely to occur if using conda.


## General usage
### Dataset generation and loading
The /datasets/ directory contains scripts for generating/downloading various image datasets.

As an example, the MNIST dataset can be downloaded -- to the directoy specified within the given scrip.py -- by running the following command:
```
$ python gen_mnist.py
```

### Training models
The provided model accepts a variety of different launch options: 
```
  --g_equiv {True, False}, 
  --self_cond {True, False}, 
  --g_input {Z2, C4, D4, Z2_K, C4_K, D4_K, Z2_S, C4_S, D4_S},
  --g_output {Z2, C4, D4, Z2_K, C4_K, D4_K, Z2_S, C4_S, D4_S},
  --diff_type {pfode, ddim} 
  --pred_type {x, eps}
  --class_cond {True, False}
  --use_fp16 {True, False} ## Not be supported for some configurations
  --sampler {euler, heun, onestep, multistep, pdm, ddpm, ddim} 
  --schedule_sampler lognormal
  --lr 0.0001
  --dropout 0.1 
  --weight_decay 0.0 
  --weight_schedule karras
  --start_ema 0.95
  --ema_rate 0.999,0.9999,0.9999432189950708
  --global_batch_size 500
  --num_channels 64  
  --num_res_blocks 1
  --channel_mult 1,2,4 
  --num_head_channels 32
  --attention_resolutions 32,16,8 
  --resblock_updown {True, False}
  --use_scale_shift_norm {True, False} 
  --data_augment 0  
  --image_size 28
  --data_dir /home/datasets/dataset
  --save_interval 1000 
  --sampling_interval 1000 
  --ref_dir /home/datasets/dataset
  --sampling_dir /home/datasets/fid_samples
  --resume_checkpoint /home/checkpoints/temp/
  ...
```

A typical launch command for training a model on the LYSTO (64x64x3 random crop) dataset is:
```
$ OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/lysto64 python train_fid.py --g_equiv True --g_input Z2_G --g_output D4_G --diff_type ddim --sampler ddim --self_cond True --data_augment 0 --eqv_reg D4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,2,2 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 32 --image_size 64 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/lysto64_random_crop --ref_dir /home/datasets/lysto64_random_crop --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/lysto64/
```

Beyond using simple comamnd line arguments we also provide config files to more easily deal with all the model paramters. The config files are located in /model_config. To make use of a config files run the following bash command
```
$ ./scripts/launch_config.sh
```


### Replicating results 
The hyperparameters used to train the model(s) in the paper are covered in Appx.D.1. of the paper. Launch commands for various dataset are also included within /scripts/launch.sh.

Models in the paper were trained using 4x Nvidia A40s. 
  + Models run on the Rotated MNIST dataset were trained for ~1-2days.
  + Models run on the LYSTO (64x64x3 random crop) dataset were trained for ~7-10days.


# Code attribution
The given implementation(s) is initially based on the github repository of [Consistency Models](https://github.com/openai/consistency_models) as well as components from the [EDM](https://github.com/NVlabs/edm) repository and [k-diffusion](https://github.com/crowsonkb/k-diffusion). We also make use of a version of the [GrouPy library](https://github.com/tscohen/GrouPy) that we modified and updated to be compliant with the current (as of release) version of pytorch, following the work done by Adam Bielski, this version can be found at [GrouPy](https://github.com/ColdestCanadian/GrouPy); however, note this library was not directly used for any of the experiments in the paper.


# Citation
```bibtex
@unpublished{LuSY:2024,
    author = {Lu, Haoye and Szabados, Spencer and Yu, Yaoliang},
    title = {Structure Preserving Diffusion Models}, 
    year = {2024},
    url = {https://arxiv.org/abs/2402.19369}, 
}
```
