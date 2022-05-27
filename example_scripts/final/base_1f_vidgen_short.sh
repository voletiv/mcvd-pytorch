#!/bin/bash

#module load python/3.8.10 StdEnv/2020 cuda/11.0 cudnn/8.0.3
#source $HOME/vidgen2/bin/activate
#rsync -avz --no-g --no-p /path/to/mask-cond-video-diffusion $SLURM_TMPDIR
#cd $SLURM_TMPDIR/mask-cond-video-diffusion

## Example
#config="kth64"
#data="/home/${user}/scratch/datasets/KTH64_h5"
#devices="0,1"
#exp=/scratch/${user}/checkpoints/my_exp
#config_mod="sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
#ckpt=60000

# Test ddpm 100
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=100 model.version=DDPM --ckpt ${ckpt} --video_gen -v videos_${ckpt}_DDPM_100_nfp_${nfp}

# Test ddim 100
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=100 model.version=DDIM --ckpt ${ckpt} --video_gen -v videos_${ckpt}_DDIM_100_nfp_${nfp}

# Test ddpm 1000
#CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=1000 model.version=DDPM --ckpt ${ckpt} --video_gen -v videos_${ckpt}_DDPM_1000_nfp_${nfp}
