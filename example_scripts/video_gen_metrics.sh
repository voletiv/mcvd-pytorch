EXP=$1
CKPT=$2
NUMFRAMESPRED=$3
PREDSPERTEST=$4
DATAPATH=$5
NAME=$6
GPU1=$7
# GPU2=$8

CUDA_VISIBLE_DEVICES=$GPU1 python main.py --config $EXP/logs/config.yml --data_path $DATAPATH --exp $EXP --ckpt $CKPT --seed 0 --video_gen -v videos_${CKPT}_${NAME}_DDPM_100_traj${PREDSPERTEST} --config_mod sampling.fvd=True model.version="DDPM" sampling.subsample=100 sampling.num_frames_pred=$NUMFRAMESPRED sampling.preds_per_test=$PREDSPERTEST sampling.max_data_iter=100000000
# CUDA_VISIBLE_DEVICES=$GPU2 python main.py --config $EXP/logs/config.yml --data_path $DATAPATH --exp $EXP --ckpt $CKPT --seed 0 --video_gen -v videos_${CKPT}_${NAME}_DDIM_100_traj${PREDSPERTEST} --config_mod sampling.fvd=True model.version="DDIM" sampling.subsample=100 sampling.num_frames_pred=$NUMFRAMESPRED sampling.preds_per_test=$PREDSPERTEST sampling.max_data_iter=100000000 &
