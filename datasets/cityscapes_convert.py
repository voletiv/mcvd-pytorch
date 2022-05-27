import argparse
import cv2
import glob
import numpy as np
import os
import sys

from functools import partial
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from h5 import HDF5Maker


def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h//2 - new_h//2, h//2 + new_h//2
    c_min, c_max = w//2 - new_w//2, w//2 + new_w//2
    return image[r_min:r_max, c_min:c_max, :]


def read_video(video_files, image_size):
    frames = []
    for file in video_files:
        frame = cv2.imread(file)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cc = center_crop(img)
        pil_im = Image.fromarray(img_cc)
        pil_im_rsz = pil_im.resize((image_size, image_size), Image.LANCZOS)
        frames.append(np.array(pil_im_rsz))
    return frames


def filename_to_num(filename):
    return 1000.*sum([ord(x) for x in os.path.basename(filename).split('_')[0]]) + 100.*int(os.path.basename(filename).split('_')[1]) + int(os.path.basename(filename).split('_')[2])


def process_video(video_files, image_size):
    frames = []
    try:
        frames = read_video(video_files, image_size)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        # break
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    return frames


def make_h5_from_cityscapes_multi(cityscapes_dir, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    filenames_all = sorted(glob.glob(os.path.join(cityscapes_dir, '*', '*.png')))
    videos = np.array(filenames_all).reshape(-1, 30)

    p_video = partial(process_video, image_size=image_size)

    # Process videos 100 at a time
    pbar = tqdm(total=len(videos))
    for i in range(int(np.ceil(len(videos)/100))):

        # pool
        with Pool() as pool:
            # tic = time.time()
            frames_all = pool.imap(p_video, videos[i*100:(i+1)*100])
            # add frames to h5
            for frames in frames_all:
                if len(frames) > 0:
                    h5_maker.add_data(frames, dtype='uint8')
            # toc = time.time()

        pbar.update(len(videos[i*100:(i+1)*100]))

    pbar.close()
    h5_maker.close()


def make_h5_from_cityscapes(cityscapes_dir, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    filenames_all = sorted(glob.glob(os.path.join(cityscapes_dir, '*', '*.png')))

    videos = np.array(filenames_all).reshape(-1, 30)

    for video_files in tqdm(videos):

        try:
            frames = read_video(video_files, image_size)
            h5_maker.add_data(frames, dtype='uint8')

        except StopIteration:
            break

        except (KeyboardInterrupt, SystemExit):
            print("Ctrl+C!!")
            break

        except:
            e = sys.exc_info()[0]
            print("ERROR:", e)

    h5_maker.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--leftImg8bit_sequence_dir', type=str, help="Path to 'leftImg8bit_sequence' ")
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_cityscapes_multi(out_dir=os.path.join(args.out_dir, "train"), cityscapes_dir=os.path.join(args.leftImg8bit_sequence_dir, "train"), image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
    make_h5_from_cityscapes_multi(out_dir=os.path.join(args.out_dir, "val"), cityscapes_dir=os.path.join(args.leftImg8bit_sequence_dir, "val"), image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
    make_h5_from_cityscapes_multi(out_dir=os.path.join(args.out_dir, "test"), cityscapes_dir=os.path.join(args.leftImg8bit_sequence_dir, "test"), image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
