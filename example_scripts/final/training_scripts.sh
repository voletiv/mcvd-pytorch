
#########################################################################
############ Training Masked Conditional Diffusion Models ###############
#########################################################################

# For ease of use using sbatch, we recommend doing something like this (example from our clusters):
# sbatch --time=14:0:0 --account=def-bob --ntasks=1 --gres=gpu:v100l:1 --cpus-per-task=8 --mail-user=my_name_is_skrillex@edm.com --mail-type=ALL --mem=32G -o /scratch/skrillex/logs/vidgen_${exp}_%j.out --export=config="$config",data="$data",exp="$exp",config_mod="$config_mod",devices="$devices" base_1f.sh
# instead of
# python base_1f_vidgen_short.sh

# base_1f.sh, base_1f_2.sh, and base_1f_4.sh are for 1, 2, and 4 GPUs respectively because there seem to be a bug when we directly input "device" as a bash variables
# Everything here should fit with <= 4 V-100 with 32Gb of RAM each
# For base_1f.sh, base_1f_2.sh, and base_1f_4.sh you need these exported bash variables: config, data, devices, exp, config_mod

# your data folder should look like this:
## BAIR_h5 Cityscapes128_h5 KTH64_h5 MNIST UCF101_64.hdf5

###############
## Arguments ##
###############

## Please change the directories below to your own
export user="skrillex"
export project_dir="mask-cond-video-diffusion"
export code_folder="/home/${user}/my_projects/${project_dir}" # code folders
export logs_folder="/scratch/${user}/Output1/Extra/logs" # where to output logs
export data_folder="/home/${user}/scratch/datasets"
export exp_folder=/scratch/${user}/checkpoints

export dir="${code_folder}"
cd ${dir}

#############
## SMMNIST ##
#############

export config="smmnist_DDPM_big5"
export data="${data_folder}"
export devices="0"

# Video prediction non-spade
export exp="smmnist_big_5c5_unetm_b2"
export config_mod="training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f.sh

# Video generation non-spade
export exp="smmnist_big_5c5_unetm_b2_pmask50"
export config_mod="training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f.sh

# Video prediction spade
export exp="SMMNIST_big_c5t5_SPADE"
export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f.sh

# Video generation spade
export exp="smmnist_big_5c5_unetm_b2_spade_pmask50"
export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f.sh

# Video interpolation
export exp="smmnist_interp_big_c5t5f5_SPADE"
export config_mod="data.num_frames_future=5 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f.sh
export exp="smmnist_interp_big_c5t10f5_SPADE"
export config_mod="data.num_frames_future=5 data.num_frames=10 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f.sh

# Video interpolation and prediction spade
export exp="smmnist_big_5c5f5_unetm_b2_pmask50_future"
export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f.sh
export exp="smmnist_PredPlusInterp_big_c5t5f5_SPADE"
export config_mod="model.spade=True data.prob_mask_future=0.5 data.num_frames_future=5 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

# General
export exp="smmnist_64_5c5f5_unetm_b2_pmask50_futurepast"
export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f.sh

export exp="smmnist_PredPlusInterpPlusGen_big_c5t5f5_SPADE"
export config_mod="model.spade=True data.prob_mask_future=0.5 data.num_frames_future=5 training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f_vidgen.sh

#########
## KTH ##
#########

export config="kth64_big"
export data="${data_folder}/KTH64_h5"
export devices="0"

# Video prediction non-spade
export exp="kth64_big_5c10_unetm_b2"
export config_mod="training.snapshot_freq=50000 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_1f.sh

# Video prediction spade
export exp="kth64_verybigbig_5c10_unetm_b2_spade"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.spade=True model.spade_dim=192 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# Video interpolation+pred spade
export exp="kth64_interp_big_c10t10f5_SPADE"
export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=50000 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=10 data.num_frames_cond=10 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="kth64_interp_big_c5t5f5_SPADE"
export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=50000 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=5 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# General
export exp="kth64_big_5c5f5_futmask50_general_unetm_b2"
export config_mod="training.snapshot_freq=50000 data.prob_mask_cond=0.50 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=5 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh


##########
## BAIR ##
##########

export config="bair_big"
export data="${data_folder}/BAIR_h5"
export devices="0"

# Video prediction
export exp="bair64_big192_5c1_unetm"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="bair64_big192_5c1_pmask50_unetm"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

export exp="bair64_big192_5c2_unetm"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="bair64_big192_5c2_pmask50_unetm"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

export exp="bair64_big192_5c1_unetm_spade"
export config_mod="training.snapshot_freq=50000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="bair64_big192_5c1_pmask50_unetm_spade"
export config_mod="training.snapshot_freq=50000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

export exp="bair64_big192_5c2_unetm_spade"
export config_mod="training.snapshot_freq=50000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="bair64_big192_5c2_pmask50_unetm_spade"
export config_mod="training.snapshot_freq=50000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# General
export exp="bair64_big192_5c1_prevfutpmask50_unetm_general"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="bair64_big192_5c2_prevfutpmask50_unetm_general"
export config_mod="training.snapshot_freq=50000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 data.num_frames_future=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

################
## Cityscapes ##
################

export config="cityscapes_big"
export data="${data_folder}/Cityscapes128_h5"
export devices="0,1,2,3"

# Video prediction
export exp="city32_big192_5c2_unetm_long"
export config_mod="model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2  training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
export exp="city16_big128_256_5c2_unetm_long_spade"
export config_mod="model.spade=True model.spade_dim=128 model.ngf=256 model.n_head_channels=256 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2  training.batch_size=16 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# video prediction and interpolation
export exp="city16_big128_256_5c2f2_unetm_long_spade_future_maskcond"
export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.5 data.prob_mask_future=0.0 model.ngf=256 model.n_head_channels=256 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_future=2 data.num_frames_cond=2  training.batch_size=16 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# General
export exp="city16_big128_256_5c2f2_unetm_long_spade_future_maskcondfut"
export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.5 data.prob_mask_future=0.5 model.ngf=256 model.n_head_channels=256 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_future=2 data.num_frames_cond=2  training.batch_size=16 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

#############
## UCF-101 ##
#############

export config="ucf101"
export data="${data_folder}"
export devices="0,1,2,3"
export nfp="16"

# Video prediction
exp="ucf10132_big288_4c4_unetm"
config_mod="model.ngf=288 model.n_head_channels=288 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
exp="ucf10132_big192_288_4c4_unetm_spade"
config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# Video generation
exp="ucf10132_big288_4c4_pmask50_unetm"
config_mod="model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
exp="ucf10132_big192_288_4c4_pmask50_unetm_spade"
config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh

# General
exp="ucf10132_big192_288_4c4f4_unetm_spade_pmask50cond_spade"
config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.prob_mask_future=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 data.num_frames_future=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
exp="ucf10132_big192_288_4c4f4_pmask50condfut_unetm_spade"
config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 data.prob_mask_future=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_future=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
sh ./example_scripts/final/base_4f.sh
