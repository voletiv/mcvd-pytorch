
######################################################################
############ Sampling DDPM-1000, DDPM-100 and DDIM-100 ###############
######################################################################

# For ease of use using sbatch, we recommend doing something like this (example from our clusters):
# sbatch --time=14:0:0 --account=def-bob --ntasks=1 --gres=gpu:v100l:1 --cpus-per-task=8 --mail-user=my_name_is_skrillex@edm.com --mail-type=ALL --mem=32G -o /scratch/${user}/logs/vidgen_ckpt${ckpt}_nfp${nfp}_${exp}_%j.out --export=config="$config",data="$data",exp="$exp",config_mod="$config_mod",devices="$devices",ckpt="$ckpt",nfp="$nfp" base_1f_vidgen_short.sh
# instead of
# python base_1f_vidgen_short.sh

# Warning: DDPM-1000 is super slow, we did not do it for most models, for faster speed use "base_1f_vidgen_short.sh" instead of "base_1f_vidgen.sh"
# Everything here should fit within a single V-100 with 32Gb of RAM
# For base_1f_vidgen.sh and base_1f_vidgen_short.sh, you need these exported bash variables: config, data, devices, exp, config_mod, ckpt, nfp

# your data folder should look like this:
## BAIR_h5 Cityscapes128_h5 KTH64_h5 MNIST UCF101_64.hdf5

## All checkpoints are available here: https://drive.google.com/drive/folders/15pDq2ziTv3n5SlrGhGM0GVqwIZXgebyD?usp=sharing

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
export nfp="10"

export exp="smmnist_big_5c5_unetm_b2"
export exp=${exp_folder}/${exp}
export ckpt="700000"
export config_mod="sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="SMMNIST_big_c5t5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="140000"
export config_mod="model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="smmnist_interp_big_c5t5f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="150000"
export config_mod="model.spade=True data.num_frames_future=5 model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="smmnist_interp_big_c5t10f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="125000"
export config_mod="model.spade=True data.num_frames_future=5 data.num_frames=10 model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="smmnist_PredPlusInterp_big_c5t5f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="125000"
export config_mod="data.prob_mask_future=0.5 model.spade=True data.num_frames_future=5 data.num_frames=5 model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="smmnist_PredPlusInterpPlusGen_big_c5t5f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="375000"
export config_mod="data.prob_mask_future=0.5 data.prob_mask_cond=0.50 model.spade=True data.num_frames_future=5 data.num_frames=5 model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="smmnist_64_5c5f5_unetm_b2_pmask50_futurepast"
export exp=${exp_folder}/${exp}
export ckpt="400000"
export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="smmnist_64_5c5f5_unetm_b2_pmask50_futurepast"
export exp=${exp_folder}/${exp}
export ckpt="550000"
export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="smmnist_64_5c5f5_unetm_b2_pmask50_futurepast"
export exp=${exp_folder}/${exp}
export ckpt="650000"
export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
sh ./example_scripts/final/base_1f_vidgen.sh

#########
## KTH ##
#########

export config="kth64_big"
export data="${data_folder}/KTH64_h5"
export devices="0"

export exp="kth64_big_5c10_unetm_b2"
export exp=${exp_folder}/${exp}
export ckpt="400000"
export nfp="30"
export config_mod="data.num_frames=5 data.num_frames_future=0 data.num_frames_cond=10 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="kth64_verybigbig_5c10_unetm_b2_spade"
export exp=${exp_folder}/${exp}
export ckpt="340000"
export nfp="30"
export config_mod="model.spade=True model.spade_dim=192 model.ngf=192 model.n_head_channels=96 data.num_frames=5 data.num_frames_future=0 data.num_frames_cond=10 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh
export ckpt="480000"
export nfp="30"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="kth64_interp_big_c10t10f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="125000"
export nfp="30"
export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 data.num_frames=10 data.num_frames_future=5 data.num_frames_cond=10 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="kth64_interp_big_c5t5f5_SPADE"
export exp=${exp_folder}/${exp}
export ckpt="375000"
export nfp="30"
export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 data.num_frames=5 data.num_frames_future=5 data.num_frames_cond=5 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="kth64_big_5c5f5_futmask50_general_unetm_b2"
export exp=${exp_folder}/${exp}
export ckpt="300000"
export nfp="30"
export config_mod="data.prob_mask_cond=0.50 data.prob_mask_future=0.50 data.num_frames=5 data.num_frames_future=5 data.num_frames_cond=5 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="40"
sh ./example_scripts/final/base_1f_vidgen.sh

##########
## BAIR ##
##########
# Note that our SPATIN models here have early checkpoints as there was a cluster maintenance :(

export config="bair_big"
export data="${data_folder}/BAIR_h5"
export devices="0"

export exp="bair64_big192_5c1_unetm"
export exp=${exp_folder}/${exp}
export ckpt="750000"
export nfp="15"
export config_mod="model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c1_pmask50_unetm"
export exp=${exp_folder}/${exp}
export ckpt="650000"
export config_mod="data.prob_mask_cond=0.50 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c2_unetm"
export exp=${exp_folder}/${exp}
export ckpt="450000"
export nfp="28"
export config_mod="model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c2_pmask50_unetm"
export exp=${exp_folder}/${exp}
export ckpt="650000"
export config_mod="data.prob_mask_cond=0.50 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c1_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="210000"
export nfp="15"
export config_mod="training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c1_pmask50_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="170000"
export config_mod="data.prob_mask_cond=0.50 training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c2_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="390000"
export nfp="28"
export config_mod="training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c2_pmask50_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="270000"
export config_mod="data.prob_mask_cond=0.50 training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c2_unetm"
export exp=${exp_folder}/${exp}
export ckpt="450000"
export nfp="14"
export config_mod="model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c2_pmask50_unetm"
export exp=${exp_folder}/${exp}
export ckpt="650000"
export config_mod="data.prob_mask_cond=0.50 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c2_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="390000"
export nfp="14"
export config_mod="training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export exp="bair64_big192_5c2_pmask50_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="270000"
export config_mod="data.prob_mask_cond=0.50 training.sample_freq=5000 training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh


export exp="bair64_big192_5c1_prevfutpmask50_unetm_general"
export exp=${exp_folder}/${exp}
export ckpt="530000"
export nfp="15"
export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=2 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export ckpt="180000"
export nfp="15"
export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=2 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh

export exp="bair64_big192_5c2_prevfutpmask50_unetm_general"
export exp=${exp_folder}/${exp}
export ckpt="490000"
export nfp="14"
export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=2 data.num_frames_future=2 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh
export nfp="28"
export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=2 data.num_frames_future=2 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen.sh


################
## Cityscapes ##
################

export config="cityscapes_big"
export data="${data_folder}/Cityscapes128_h5"
export devices="0"

exp="city32_big192_5c2_unetm_long"
export exp=${exp_folder}/${exp}
ckpt="900000"
config_mod="model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=32 sampling.batch_size=35 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh

exp="city16_big128_256_5c2_unetm_long_spade"
export exp=${exp_folder}/${exp}
ckpt="650000"
config_mod="model.spade=True model.spade_dim=128 model.ngf=256 model.n_head_channels=256 data.num_frames=5 data.num_frames_cond=2 sampling.batch_size=45 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh

#############
## UCF-101 ##
#############

export config="ucf101"
export data="${data_folder}"
export devices="0"
export nfp="16"

export exp="ucf10132_big288_4c4_unetm"
export exp=${exp_folder}/${exp}
export ckpt="1050000"
export config_mod="model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh
export exp="ucf10132_big288_4c4_pmask50_unetm"
export exp=${exp_folder}/${exp}
export ckpt="900000"
export config_mod="data.prob_mask_cond=0.50 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh

export exp="ucf10132_big192_288_4c4_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="550000"
export config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh
export exp="ucf10132_big192_288_4c4_pmask50_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="650000"
export config_mod="data.prob_mask_cond=0.50 model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh

export exp="ucf10132_big192_288_4c4_pmask50_unetm_spade"
export exp=${exp_folder}/${exp}
export ckpt="250000"
export config_mod="data.prob_mask_cond=0.50 model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh