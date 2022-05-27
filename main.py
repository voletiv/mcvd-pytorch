import argparse
import copy
import datetime
import glob
import logging
import numpy as np
import os
import shutil
import sys
import torch
import traceback
import time
import yaml
# import torch.utils.tensorboard as tb
# from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)

from runners import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', required=True, help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')

    parser.add_argument('--test', action='store_true', help='Whether to test the model')

    parser.add_argument('--feats_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets'),
                            help='Path to directory containing InceptionV3 feats pt files')
    parser.add_argument('--stats_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets'),
                            help='Path to directory containing fid_stats npz files')
    parser.add_argument('--stats_download', action='store_true', help='Whether to download fid stats')

    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--fid_batch_size', type=int, default=1000, help='Batch size in InceptionNetV3 for FID calc')
    parser.add_argument('--no_pr', action='store_true', help="No PR calc, only FID calc. Generally unnecessary.")
    parser.add_argument('--fid_num_samples', type=int, default=None, help='# of samples for FID, to override config.fast_fid.num_samples, when using sample/test/fast_fid')
    parser.add_argument('--pr_nn_k', type=int, default=None, help='# of nearest neighbours for Precision/Recall, to override config.fast_fid.pr_nn_k, when using sample/test/fast_fid')

    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--final_only', type=eval, default=None, choices=[True, False], help='Whether to save ONLY final image or all sampling steps, when using sample/test/fast_fid')

    parser.add_argument('--end_ckpt', type=int, default=None, help='Model checkpoint # to load until, when using test/fast_fid')
    parser.add_argument('--freq', type=int, default=None, help='Model checkpoint freq to load, when using test/fast_fid')

    parser.add_argument('--no_ema', action='store_true', help="Don't use Exponential Moving Average")

    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--interact', action='store_true', help='Whether to interact') # basically do nothing

    ### Above are options are from the original code, below are new options for videos

    parser.add_argument('--video_gen', action='store_true', help='Whether to produce video samples from the conditional model')
    parser.add_argument('-v', '--video_folder', type=str, default='videos', help="The folder name of video samples")

    parser.add_argument('--subsample', type=int, default=None, help='# of samples in path, to override config.sampling.subsample, when using sample/test/fast_fid')
    parser.add_argument('--ckpt', type=int, default=None, help='Model checkpoint # to load from, when using sample/video_gen/test/fast_fid')

    parser.add_argument('--config_mod', nargs='*', type=str, default=[], help='Overrid config options, e.g., model.ngf=64 model.spade=True training.batch_size=32') 

    parser.add_argument('--start_at', type=int, default=0, help="For KTH, can start at Kth frame in test and ignore the rest")


    args = parser.parse_args()
    args.command = 'python ' + ' '.join(sys.argv)
    # args.log_path = os.path.join(args.exp, 'logs', args.doc)
    args.log_path = os.path.join(args.exp, 'logs')

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override with config_mod
    for val in args.config_mod:
        val, config_val = val.split('=')
        config_type, config_name = val.split('.')
        try:
            totest = config[config_type][config_name][0]
        except:
            totest = config[config_type][config_name]

        if isinstance(totest, str):
            config[config_type][config_name] = config_val
        else:
            config[config_type][config_name] = eval(config_val)

    # Override
    if config['data']['dataset'].upper() == "IMAGENET":
        if config['data']['classes'] is None:
            config['data']['classes'] = []
        elif config['data']['classes'] == 'dogs':
            config['data']['classes'] = list(range(151, 269))
        assert isinstance(config['data']['classes'], list), "config['data']['classes'] must be a list!"
    config['sampling']['subsample'] = args.subsample or config['sampling'].get('subsample')
    config['fast_fid']['batch_size'] = args.fid_batch_size or config['fast_fid']['batch_size']
    config['fast_fid']['num_samples'] = args.fid_num_samples or config['fast_fid']['num_samples']
    config['fast_fid']['pr_nn_k'] = args.pr_nn_k or config['fast_fid'].get('pr_nn_k', 3)
    if args.no_ema:
        config['model']['ema'] = False

    if config['sampling'].get('fvd', False) and config['sampling'].get('num_frames_pred', 10) < 10:
        print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when sampling.num_frames_pred < 10 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        config['sampling']['fvd'] =  False
    #if config['sampling'].get('fvd', False) and config['data']['channels'] != 3:
    #    print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when image is greyscale >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #    config['sampling']['fvd'] =  False

    if config['model'].get('output_all_frames', False): 
        noise_in_cond = True # if False, then wed predict the input-cond frames z, but the z is zero everywhere which is weird and seems irrelevant to predict. So we stick to the noise_in_cond case.

    assert not config['model'].get('cond_emb', False) or (config['model'].get('cond_emb', False) and config['data'].get('prob_mask_cond',0.0) > 0)

    if config['data'].get('prob_mask_sync', False):
        assert config['data'].get('prob_mask_cond', 0.0) > 0 and config['data'].get('prob_mask_cond', 0.0) == config['data'].get('prob_mask_future', 0.0)

    # if config['sampling'].get('preds_per_test', 1) > 1:
    #     assert config['sampling'].get('preds_per_test', 1) >= 5, f"preds_per_test can be either 1, or >=5. Got {config['sampling'].get('preds_per_test', 1)}"

    # # Override if interpolation
    # if config['data'].get('num_frames_future', 0) > 0:
    #     config['sampling']['num_frames_pred'] = config['data']['num_frames']

    new_config = dict2namespace(config)

    # tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test and not args.sample and not args.video_gen and not args.fast_fid:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(f"Folder {args.log_path} already exists.\nOverwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    # shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    # if os.path.exists(tb_path):
                    #     shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            with open(os.path.join(args.log_path, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

            # Code
            code_path = os.path.join(args.exp, 'code')
            os.makedirs(code_path, exist_ok=True)
            copy_scripts(os.path.dirname(os.path.abspath(__file__)), code_path)

        # new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:

            if args.ckpt is not None:
                new_config.sampling.ckpt_id = args.ckpt
            if new_config.sampling.ckpt_id == 0 :
                new_config.sampling.ckpt_id = None
            if args.final_only is not None:
                new_config.sampling.final_only = args.final_only

            if new_config.sampling.final_only:
                os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
                args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
            else:
                os.makedirs(os.path.join(args.exp, f'image_samples_{new_config.sampling.ckpt_id}'), exist_ok=True)
                args.image_folder = os.path.join(args.exp, f'image_samples_{new_config.sampling.ckpt_id}', args.image_folder)

            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(f"Image folder {args.image_folder} already exists.\nOverwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

            with open(os.path.join(args.image_folder, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            with open(os.path.join(args.image_folder, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)


        elif args.video_gen:

            new_config.sampling.ckpt_id = args.ckpt or new_config.sampling.ckpt_id
            args.final_only = True

            # if new_config.sampling.final_only:
            os.makedirs(os.path.join(args.exp, 'video_samples'), exist_ok=True)
            args.video_folder = os.path.join(args.exp, 'video_samples', args.video_folder)
            # else:
            #     os.makedirs(os.path.join(args.exp, f'image_samples_{new_config.sampling.ckpt_id}'), exist_ok=True)
            #     args.image_folder = os.path.join(args.exp, f'image_samples_{new_config.sampling.ckpt_id}', args.image_folder)

            if not os.path.exists(args.video_folder):
                os.makedirs(args.video_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(f"Video folder {args.video_folder} already exists.\nOverwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.video_folder)
                    os.makedirs(args.video_folder)
                else:
                    print("Output video folder exists. Program halted.")
                    sys.exit(0)

            with open(os.path.join(args.video_folder, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            with open(os.path.join(args.video_folder, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)


        elif args.fast_fid:

            new_config.fast_fid.begin_ckpt = args.ckpt or new_config.fast_fid.begin_ckpt
            new_config.fast_fid.end_ckpt = args.end_ckpt or new_config.fast_fid.end_ckpt
            new_config.fast_fid.freq = args.freq or getattr(new_config.fast_fid, "freq", 5000)

            os.makedirs(os.path.join(args.exp, 'fid_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'fid_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = False
                else:
                    response = input(f"Image folder {args.image_folder} already exists.\n "
                                     "Type Y to delete and start from an empty folder?\n"
                                     "Type N to overwrite existing folders (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)

            with open(os.path.join(args.image_folder, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            with open(os.path.join(args.image_folder, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

        elif args.test:
            new_config.test.begin_ckpt = args.ckpt or new_config.test.begin_ckpt
            new_config.test.end_ckpt = args.end_ckpt or new_config.test.end_ckpt
            new_config.test.freq = args.freq or getattr(new_config.test, "freq", 5000)

            with open(os.path.join(args.image_folder, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            with open(os.path.join(args.image_folder, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    config_uncond = new_config

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, config_uncond


def copy_scripts(src, dst):
    print("Copying scripts in", src, "to", dst)
    for file in glob.glob(os.path.join(src, '*.sh')) + \
            glob.glob(os.path.join(src, '*.py')) + \
            glob.glob(os.path.join(src, '*_means.pt')) + \
            glob.glob(os.path.join(src, '*.data')) + \
            glob.glob(os.path.join(src, '*.cfg')) + \
            glob.glob(os.path.join(src, '*.yml')) + \
            glob.glob(os.path.join(src, '*.names')):
        shutil.copy(file, dst)
    for d in glob.glob(os.path.join(src, '*/')):
        if '__' not in os.path.basename(os.path.dirname(d)) and \
                '.' not in os.path.basename(os.path.dirname(d))[0] and \
                'ipynb' not in os.path.basename(os.path.dirname(d)) and \
                os.path.basename(os.path.dirname(d)) != 'data' and \
                os.path.basename(os.path.dirname(d)) != 'experiments' and \
                os.path.basename(os.path.dirname(d)) != 'assets':
            if os.path.abspath(d) in os.path.abspath(dst):
                continue
            print("Copying", d)
            # shutil.copytree(d, os.path.join(dst, d))
            new_dir = os.path.join(dst, os.path.basename(os.path.normpath(d)))
            os.makedirs(new_dir, exist_ok=True)
            copy_scripts(d, new_dir)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, config_uncond = parse_args_and_config()
    logging.info("{}".format(args))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    # if not args.test and not args.sample and not args.fast_fid:
    #     del config_dict['tb_logger']
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)
    logging.info("Args =")
    print(">" * 80)
    args_dict = copy.copy(vars(args))
    # if not args.test and not args.sample and not args.fast_fid:
    #     del config_dict['tb_logger']
    print(yaml.dump(args_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config, config_uncond)
        if args.test:
            runner.test()
        elif args.sample:
            runner.sample()
        elif args.video_gen:
            runner.video_gen()
        elif args.fast_fid:
            runner.fast_fid()
        elif args.interact:
            pass
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    logging.info(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    return runner, args, config


if __name__ == '__main__':
    runner, args, config = main()
    if not args.interact:
        sys.exit()

# python ../main.py --config /path/to/GitHubRepos/ncsnv2-gvv/configs/cifar10_DDPM.yml --data_path /path/to/data/CIFAR10 --exp /path/to/ncsnv2/cifar10/00_DDPM_L1a_800k --comment Using L1a, unet, DDPM --seed 0 --ni

# CUDA_VISIBLE_DEVICES=2 python main.py --config configs/smmnist_DDPM_small.yaml --data_path /path/to/data/MNIST --exp /path/to/ncsnv2/SMMNIST/DDPM_small_1c5 --comment "Using L1a, unet, DDPM SMALL! Gen 1 frame conditioned on 5 frames" --seed 0 --ni
