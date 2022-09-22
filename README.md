<h1 align="center"> MCVD: Masked Conditional Video Diffusion <br/> for Prediction, Generation, and Interpolation </h1>

<h3 align="center"> <a href="https://voletiv.github.io" target="_blank">Vikram Voleti</a>*, <a href="https://ajolicoeur.wordpress.com/about/" target="_blank">Alexia Jolicoeur-Martineau</a>*, <a href="https://sites.google.com/view/christopher-pal" target="_blank">Christopher Pal</a></h3>

<h4 align="center"> NeurIPS 2022 </h4>

<h3 align="center"> <a href="https://mask-cond-video-diffusion.github.io" target="_blank">Website</a>, <a href="https://arxiv.org/abs/2205.09853" target="_blank">Paper</a>, <a href="https://ajolicoeur.wordpress.com/?p=466" target="_blank">Blog</a> </h3>

This is the official implementation of the NeurIPS 2022 paper [MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853). In this paper, we devise a general-purpose model for video prediction (forward and backward), unconditional generation, and interpolation with Masked Conditional Video Diffusion (MCVD) models. Please see our [website](https://mask-cond-video-diffusion.github.io/) for more details. This repo is based on the code from https://github.com/ermongroup/ncsnv2.

If you find the code/idea useful for your research, please cite:


```bib
@inproceedings{voleti2022MCVD,
 author = {Voleti, Vikram and Jolicoeur-Martineau, Alexia and Pal, Christopher},
 title = {MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation},
 url = {https://arxiv.org/abs/2205.09853},
 booktitle = {(NeurIPS) Advances in Neural Information Processing Systems},
 year = {2022}
}

```

## Scaling

The models from our paper were trained with 1 to 4 GPUs (requiring from 32GB to 160GB of RAM). Models can be scaled with less or more GPUs by changing the following parameters:
* model.ngf and model.n_heads_channel (doubling ngf and n_heads_channels approximately doubles the memory demand)
* model.num_res_blocks (number of sequential residual layers per block)
* model.ch_mult=[1,2,3,4,4,4] will use 6 resblocks instead of the default 4 (model.ch_mult=[1,2,3,4])
* training.batch_size (doubling the batch size approximately increase the memory demand by 50%)
* SPATIN models can be scaled through model.spade_dim (128 -> 512 increase memory demands by 2x, 128 -> 1024 increase memory demand by 4x); it should be scaled proportionally to the number of past+future frames for best results. In practice we find the SPATIN models often need very large spade_dim to be competitive, thus we recommend regular users to stick to concatenation.

## Installation

```
# if using conda (ignore otherwise)
conda create --name vid python=3.8
# # (Optional) If your machine has a GCC/G++ version < 5:
# conda install -c conda-forge gxx=8.5.0    # (should be executed before the installation of pytorch, torchvision, and torchaudio)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt # install all requirements
```

## Experiments

The experiments to reproduce the paper can be found in [/example_scripts/final/training_scripts.sh](https://github.com/voletiv/mcvd-pytorch/blob/master/example_scripts/final/training_scripts.sh) and [/example_scripts/final/sampling_scripts.sh](https://github.com/voletiv/mcvd-pytorch/blob/master/example_scripts/final/sampling_scripts.sh).

We also provide a small notebook demo for sampling from SMMNIST: https://github.com/voletiv/mcvd-pytorch/blob/master/MCVD_demo_SMMNIST.ipynb.

## Pretrained Checkpoints and results

The checkpoints used for the experiments and their results can be used here: https://drive.google.com/drive/u/1/folders/15pDq2ziTv3n5SlrGhGM0GVqwIZXgebyD

## Configurations

The models configurations are available at /configs. To overide any existing configuration from a config file, you can simply use the --config_mod argument in the command line. For example:
```
--config_mod training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2
```

The important config options are:
```
training.batch_size=64 # training batch size

sampling.batch_size=200 # sampling batch size
sampling.subsample=100 # how many diffusion steps to take (1000 is best but is slower, 100 is faster)
sampling.max_data_iter=1000 # how many mini-batches of the test to go through at the maximum (set to 1 for training and a large value for sampling)

model.ngf=192 # number of channels (controls model size)
model.n_head_channels=192 # number of channels per self-attention head (should ideally be larger or equal to model.ngf, otherwise you may have a size mismatch error)
model.spade=True # if True uses space-time adaptive normalization instead of concatenation
model.spade_dim=128 # number of channels in space-time adaptive normalization; worth increasing, especially if conditioning on a large number of frames

sampling.num_frames_pred=16 # number of frames to predict (autoregressively)
data.num_frames=4 # number of current frames
data.num_frames_cond=4 # number of previous frames
data.num_frames_future=4 # number of future frames

data.prob_mask_cond=0.50 # probability of masking the previous frames (allows predicting current frames with no past frames)
data.prob_mask_future=0.50 # probability of masking the future frames (allows predicting current frames with no future frames)
```

When `data.num_frames_future > 0`, `data.num_frames_cond > 0`, `data.prob_mask_cond=0.50`, and `data.prob_mask_future=0.50`, one can do video prediction (forward and backward), generation, and interpolation.

## Training

You can train on Stochastic Moving MNIST with 1 GPU (if memory issues, use model.ngf=64) using:
```
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/smmnist_DDPM_big5.yml --data_path /my/data/path/to/datasets --exp smmnist_cat --ni
```

Log files will be saved in `<exp>/logs/smmnist_cat`. This folder contains stdout, metric plots, and video samples over time.

You can train on Cityscapes with 4 GPUs using:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config configs/cityscapes_big_spade.yml --data_path /my/data/path/to/datasets --exp exp_city_spade --ni
```

## Sampling

You can look at stdout or the metric plots in `<exp>/logs/smmnist_cat` to determine which checkpoint provides the best metrics. Then, you can sample from 25 frames using the chosen checkpoint (e.g., 250k) of the previous SMNIST model by running `main.py` with the `--video_gen` option:
```
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/smmnist_DDPM_big5.yml --data_path /my/data/path/to/datasets --exp smmnist_cat --ni --config_mod sampling.max_data_iter=1000 sampling.num_frames_pred=25 sampling.preds_per_test=10 sampling.subsample=100 model.version=DDPM --ckpt 250000 --video_gen -v videos_250k_DDPM_1000_nfp_pred25
```

Results will be saved in `<exp>/video_samples/videos_250k_DDPM_1000_nfp_pred25`.

You can use the above option to sample videos from any pretrained MCVD model.

## Esoteric options

We tried a few options that did not help, but we left them in the code. Some of these options might be broken, we make no guarantees, use them at your own risk. 
```
model.gamma=True # Gamma noise from https://arxiv.org/abs/2106.07582
training.L1=True # L1 loss
model.cond_emb=True # Embedding for wether we mask (1) or we don't mask (0)
output_all_frames=True # Option to output/predict all frames, not just current frames
noise_in_cond=True # Diffusion noise also in conditioning frames
one_frame_at_a_time=True # Autoregressive one image at a time instead of blockwise
model.version=FPNDM # F-PNDM from https://arxiv.org/abs/2202.09778
```

Note that this code can be used to generate images by setting data.num_frames=0, data.num_frames_cond=0, data.num_frames_future=0. 

Many unused options also exist which are from the original code by https://github.com/ermongroup/ncsnv2, mostly applicable only to images.

## For LPIPS

The code will do it for you!
> Code will download https://download.pytorch.org/models/alexnet-owt-7be5be79.pth and move it into: `models/weights/v0.1/alex.pth`

## For FVD

The code will do it for you!

> Code will download i3D model pretrained on Kinetics-400 from "https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI"
> Use `models/fvd/convert_tf_pretrained.py` to make `i3d_pretrained_400.pt`


# Datasets

## Stochastic Moving MNIST (64x64, ch1)

The script will automatically download the PyTorch MNIST dataset, which will be used to generate Stochastic Moving MNIST dynamically.

## KTH (64x64, ch1)

Download the hdf5 dataset:
```
gdown --fuzzy https://drive.google.com/file/d/1d2UfHV6RhSrwdDAlCFY3GymtFPpmh_8X/view?usp=sharing
```

> **How the data was processed:**
> 1. Download KTH dataset to `/path/to/KTH`:\
> `sh kth_download.sh /path/to/KTH`
> 2. Convert 64x64 images to HDF5 format:\
> `python datasets/kth_convert.py --kth_dir '/path/to/KTH' --image_size 64 --out_dir '/path/to/KTH64_h5' --force_h5 False`

## BAIR (64x64, ch3)

Download the hdf5 dataset:
```
gdown --fuzzy https://drive.google.com/file/d/1-R_srAOy5ZcylGXVernqE4WLCe6N4_wq/view?usp=sharing
```

> **How the data was processed:**
> 1. Download BAIR Robotic Push dataset to `/path/to/BAIR`:\
> `sh bair_dowload.sh /path/to/BAIR`
> 2. Convert it to HDF5 format, and save in `/path/to/BAIR_h5`:\
> `python datasets/bair_convert.py --bair_dir '/path/to/BAIR' --out_dir '/path/to/BAIR_h5'`


## Cityscapes (64x64, ch3)

```
gdown --fuzzy https://drive.google.com/file/d/1oP7n-FUfa9ifsMn6JHNS9depZfftvrXx/view?usp=sharing
```

> **How the data was processed:**\
> MAKE SURE YOU HAVE ~657GB SPACE! 324GB for the zip file, and 333GB for the unzipped image files
> 1. Download Cityscapes video dataset (`leftImg8bit_sequence_trainvaltest.zip` (324GB)) :\
> `sh cityscapes_download.sh username password`\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; using your `username` and `password` that you created on https://www.cityscapes-dataset.com/
> 2. Convert it to HDF5 format, and save in `/path/to/Cityscapes<image_size>_h5`:\
> `python datasets/cityscapes_convert.py --leftImg8bit_sequence_dir '/path/to/Cityscapes/leftImg8bit_sequence' --image_size 64 --out_dir '/path/to/Cityscapes64_h5'`

## Cityscapes (128x128, ch3)

Download the hdf5 dataset:
```
gdown --fuzzy https://drive.google.com/file/d/13yaJkKtmDsgtaEvuXKSvbix5usea6TJy/view?usp=sharing
```

> **How the data was processed:**\
> MAKE SURE YOU HAVE ~657GB SPACE! 324GB for the zip file, and 333GB for the unzipped image files
> 1. Download Cityscapes video dataset (`leftImg8bit_sequence_trainvaltest.zip` (324GB)) :\
> `sh cityscapes_download.sh /path/to/download/to username password`\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; using your `username` and `password` that you created on https://www.cityscapes-dataset.com/
> 2. Convert it to HDF5 format, and save in `/path/to/Cityscapes<image_size>_h5`:\
> `python datasets/cityscapes_convert.py --leftImg8bit_sequence_dir '/path/to/Cityscapes/leftImg8bit_sequence' --image_size 128 --out_dir '/path/to/Cityscapes128_h5'`

## UCF-101 (orig:320x240, ch3)

Download the hdf5 dataset:
```
gdown --fuzzy https://drive.google.com/file/d/1bDqhhfKYrdbIIOZeJcWHWjSyFQwmO1t-/view?usp=sharing
```

> **How the data was processed:**\
> MAKE SURE YOU HAVE ~20GB SPACE! 6.5GB for the zip file, and 8GB for the unzipped image files
> 1. Download UCF-101 video dataset (`UCF101.rar` (6.5GB)) :\
> `sh cityscapes_download.sh /download/dir`\
> 2. Convert it to HDF5 format, and save in `/path/to/UCF101_h5`:\
> `python datasets/ucf101_convert.py --out_dir /path/to/UCF101_h5 --ucf_dir /download/dir/UCF-101 --splits_dir /download/dir/ucfTrainTestlist`
