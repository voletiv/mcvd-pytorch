import argparse
import cv2
import glob
import imageio
import numpy as np
import os
import sys

from functools import partial
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from h5 import HDF5Maker


class UCF101_HDF5Maker(HDF5Maker):

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('data')
        self.writer.create_group('target')

    def add_video_data(self, data, dtype=None):
        data, target = data
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['target'].create_dataset(str(self.count), data=target, dtype='uint8')
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")


def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h//2 - new_h//2, h//2 + new_h//2
    c_min, c_max = w//2 - new_w//2, w//2 + new_w//2
    return image[r_min:r_max, c_min:c_max, :]


def read_video(video_file, image_size):
    frames = []
    reader = imageio.get_reader(video_file)
    h, w = 240, 320
    new_h = image_size
    new_w = int(new_h / h * w)
    for img in reader:
        # img_cc = center_crop(img)
        pil_im = Image.fromarray(img)
        pil_im_rsz = pil_im.resize((new_w, new_h), Image.LANCZOS)
        frames.append(np.array(pil_im_rsz))
        # frames.append(np.array(img))
    return np.stack(frames)


def process_video(video_file, image_size):
    frames = []
    try:
        frames = read_video(video_file, image_size)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        return "break"
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    return frames


def read_splits(splits_dir, split_idx, ucf_dir):
    # train
    txt_train = os.path.join(splits_dir, f"trainlist0{split_idx}.txt")
    vids_train = open(txt_train, 'r').read().splitlines()
    vids_train = [os.path.join(ucf_dir, line.split('.avi')[0] + '.avi') for line in vids_train]
    # test
    txt_test = os.path.join(splits_dir, f"testlist0{split_idx}.txt")
    vids_test = open(txt_test, 'r').read().splitlines()
    vids_test = [os.path.join(ucf_dir, line) for line in vids_test]
    # classes
    classes = {line.split(' ')[-1]: int(line.split(' ')[0])-1 for line in open(os.path.join(splits_dir, 'classInd.txt'), 'r').read().splitlines()}
    classes_train = [classes[os.path.basename(os.path.dirname(f))] for f in vids_train]
    classes_test = [classes[os.path.basename(os.path.dirname(f))] for f in vids_test]
    return vids_train, vids_test, classes_train, classes_test


# def make_h5_from_ucf_multi(ucf_dir, splits_dir, split_idx, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

#     # H5 maker
#     h5_maker = UCF101_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

#     vids_train, vids_test, classes_train, classes_test = read_splits(splits_dir, split_idx, ucf_dir)
#     print("Train:", len(vids_train), "\nTest", len(vids_test))

#     h5_maker.writer.create_dataset('num_train', data=len(vids_train))
#     h5_maker.writer.create_dataset('num_test', data=len(vids_test))
#     videos = vids_train + vids_test
#     classes = classes_train + classes_test

#     # Process videos 100 at a time
#     pbar = tqdm(total=len(videos))
#     for i in range(int(np.ceil(len(videos)/100))):

#         # pool
#         with Pool() as pool:
#             # tic = time.time()
#             results = pool.imap(process_video, [(v, c) for v, c in zip(videos[i*100:(i+1)*100], classes[i*100:(i+1)*100])])
#             # add frames to h5
#             for result in results:
#                 frames, t = result
#                 if len(frames) > 0:
#                     h5_maker.add_data(result, dtype='uint8')
#             # toc = time.time()

#         pbar.update(len(videos[i*100:(i+1)*100]))

#     pbar.close()
#     h5_maker.close()


def make_h5_from_ucf(ucf_dir, splits_dir, split_idx, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = UCF101_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    vids_train, vids_test, classes_train, classes_test = read_splits(splits_dir, split_idx, ucf_dir)
    print("Train:", len(vids_train), "\nTest", len(vids_test))

    h5_maker.writer.create_dataset('num_train', data=len(vids_train))
    h5_maker.writer.create_dataset('num_test', data=len(vids_test))
    videos = vids_train + vids_test
    classes = classes_train + classes_test

    for i in tqdm(range(len(videos))):
        frames = process_video(videos[i], image_size)
        if isinstance(frames, str) and frames == "break":
            break
        h5_maker.add_data((frames, classes[i]), dtype='uint8')

    h5_maker.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--ucf_dir', type=str, help="Path to UCF-101 videos")
    parser.add_argument('--splits_dir', type=str, help="Path to ucfTrainTestlist")
    parser.add_argument('--split_idx', type=int, choices=[1, 2, 3], default=3, help="Which split to use")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_ucf(out_dir=args.out_dir, ucf_dir=args.ucf_dir, splits_dir=args.splits_dir, split_idx=args.split_idx,
                     image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
