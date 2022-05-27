import argparse
import glob
import imageio
import numpy as np
import os
import pickle
import sys
import tarfile

from functools import partial
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from h5 import HDF5Maker


class Kinetics_HDF5Maker(HDF5Maker):

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('target')

    def add_video_data(self, data, dtype='uint8'):
        data, target = data
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['target'].create_dataset(str(self.count), data=target)
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")


def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h//2 - new_h//2, h//2 + new_h//2
    c_min, c_max = w//2 - new_w//2, w//2 + new_w//2
    return image[r_min:r_max, c_min:c_max, :]


def read_video(video, image_size):
    frames = []
    reader = imageio.get_reader(video)
    for img in reader:
        img_cc = center_crop(img)
        pil_im = Image.fromarray(img_cc)
        pil_im_rsz = pil_im.resize((image_size, image_size), Image.LANCZOS)
        frames.append(np.array(pil_im_rsz))
    # return
    return frames


def read_video2(video, image_size1, image_size2):
    frames1 = []
    frames2 = []
    reader = imageio.get_reader(video)
    for img in reader:
        img_cc = center_crop(img)
        pil_im = Image.fromarray(img_cc)
        pil_im_rsz1 = pil_im.resize((image_size1, image_size1), Image.LANCZOS)
        pil_im_rsz2 = pil_im.resize((image_size2, image_size2), Image.LANCZOS)
        frames1.append(np.array(pil_im_rsz1))
        frames2.append(np.array(pil_im_rsz2))
    # return
    return (frames1, frames2)


def append_to_dict_list(d, key, value):
    if key not in d:
        d[key] = []
    d[key].append(value)


def process_video(v, targz_file, out_dir, image_size, image_size2=None):
    frames = []
    if image_size2 is not None:
        frames = ([], [])
    # extract mp4 video
    with tarfile.open(targz_file) as file:
        file.extract(v, out_dir)
    # Add to dataset
    try:
        if image_size2 is None:
            frames = read_video(os.path.join(out_dir, os.path.basename(v)), image_size)
        else:
            frames = read_video2(os.path.join(out_dir, os.path.basename(v)), image_size, image_size2)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        # break
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    # Remove mp4 video
    os.remove(os.path.join(out_dir, os.path.basename(v)))
    return frames


# Make 2 HDF5s simultaneously with different image_sizes from Kinetics dataset using multiprocessing
def make_h5_from_kinetics_multi2(kinetics600_dir, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False, test=False,
                                 videos_per_class=None, video_idx_per_class=0, image_size2=None, out_dir2=None):

    # H5 maker
    h5_maker = Kinetics_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    if image_size2 is not None:
        h5_maker2 = Kinetics_HDF5Maker(out_dir2, num_per_shard=vids_per_shard, force=force_h5, video=True)

    targz_all = sorted(glob.glob(os.path.join(kinetics600_dir, '*.tar.gz')))

    targets = {}
    target_names = []
    prev_count = 0
    for target, targz_file in tqdm(enumerate(targz_all), total=len(targz_all)):

        t = target if not test else -1
        if not test:
            target_names.append(os.path.splitext(os.path.splitext(os.path.basename(targz_file))[0])[0])

        # get video filenames
        with tarfile.open(targz_file) as file:
            video_files = [v for v in sorted(file.getnames()) if os.path.splitext(v)[-1] == '.mp4']

        if videos_per_class is not None:
            video_files = video_files[video_idx_per_class:video_idx_per_class+videos_per_class]

        # process videos
        p_video = partial(process_video, targz_file=targz_file, out_dir=out_dir, image_size=image_size, image_size2=image_size2)

        # process 100 videos at a time
        pbar = tqdm(total=len(video_files))
        for i in range(int(np.ceil(len(video_files)/100))):

            # pool
            with Pool() as pool:
                # tic = time.time()
                frames_all = pool.imap(p_video, video_files[i*100:(i+1)*100])
                # add frames to h5
                for frames in frames_all:
                    if image_size2 is None:
                        if len(frames) > 0:
                            h5_maker.add_data((frames, t), dtype='uint8')
                    else:
                        frames, frames2 = frames
                        if len(frames) > 0:
                            h5_maker.add_data((frames, t), dtype='uint8')
                        if len(frames2) > 0:
                            h5_maker2.add_data((frames2, t), dtype='uint8')
                # toc = time.time()

            pbar.update(len(video_files[i*100:(i+1)*100]))

        pbar.close()

        # Add targets
        if not test:
            targets[target] = np.arange(prev_count, h5_maker.count).tolist()
            prev_count = h5_maker.count

        # Save targets
        if not test:
            # print("\nWriting", os.path.join(out_dir, 'targets.pkl'))
            with open(os.path.join(out_dir, 'targets.pkl'), 'wb') as f:
                pickle.dump(targets, f)

            # Save target_names
            # print("Writing", os.path.join(out_dir, 'target_names.txt\n'))
            with open(os.path.join(out_dir, 'target_names.txt'), 'w') as f:
                f.writelines(target_names)

            if image_size2 is not None:
                # print("\nWriting", os.path.join(out_dir2, 'targets.pkl'))
                with open(os.path.join(out_dir2, 'targets.pkl'), 'wb') as f:
                    pickle.dump(targets, f)

                # Save target_names
                # print("Writing", os.path.join(out_dir2, 'target_names.txt\n'))
                with open(os.path.join(out_dir2, 'target_names.txt'), 'w') as f:
                    f.writelines(target_names)

    # close h5
    h5_maker.close()

    if image_size2 is not None:
        h5_maker2.close()


# Make HDF5 from Kinetics dataset using multiprocessing
def make_h5_from_kinetics_multi(kinetics600_dir, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False, test=False,
                                videos_per_class=None, video_idx_per_class=0):

    # H5 maker
    h5_maker = Kinetics_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    targz_all = sorted(glob.glob(os.path.join(kinetics600_dir, '*.tar.gz')))

    targets = {}
    target_names = []
    prev_count = 0
    for target, targz_file in tqdm(enumerate(targz_all), total=len(targz_all)):

        t = target if not test else -1
        if not test:
            target_names.append(os.path.splitext(os.path.splitext(os.path.basename(targz_file))[0])[0])

        # get video filenames
        with tarfile.open(targz_file) as file:
            video_files = [v for v in sorted(file.getnames()) if os.path.splitext(v)[-1] == '.mp4']

        if videos_per_class is not None:
            video_files = video_files[video_idx_per_class:video_idx_per_class+videos_per_class]

        # process videos
        p_video = partial(process_video, targz_file=targz_file, out_dir=out_dir, image_size=image_size)

        # process 100 videos at a time
        pbar = tqdm(total=len(video_files))
        for i in range(int(np.ceil(len(video_files)/100))):

            # pool
            with Pool() as pool:
                # tic = time.time()
                frames_all = pool.imap(p_video, video_files[i*100:(i+1)*100])
                # add frames to h5
                for frames in frames_all:
                    if len(frames) > 0:
                        h5_maker.add_data((frames, t), dtype='uint8')
                # toc = time.time()

            pbar.update(len(video_files[i*100:(i+1)*100]))

        pbar.close()

        # Add targets
        if not test:
            targets[target] = np.arange(prev_count, h5_maker.count).tolist()
            prev_count = h5_maker.count

        # Save targets
        if not test:
            # print("\nWriting", os.path.join(out_dir, 'targets.pkl'))
            with open(os.path.join(out_dir, 'targets.pkl'), 'wb') as f:
                pickle.dump(targets, f)

            # Save target_names
            # print("Writing", os.path.join(out_dir, 'target_names.txt\n'))
            with open(os.path.join(out_dir, 'target_names.txt'), 'w') as f:
                f.writelines(target_names)

    # close h5
    h5_maker.close()


# Make HDF5 from Kinetics dataset
def make_h5_from_kinetics(kinetics600_dir, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False, test=False):

    # H5 maker
    h5_maker = Kinetics_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    targz_all = sorted(glob.glob(os.path.join(kinetics600_dir, '*.tar.gz')))

    targets = {}
    target_names = []
    count = 0
    for target, targz_file in tqdm(enumerate(targz_all), total=len(targz_all)):
        t = target if not test else -1
        if not test:
            target_names.append(os.path.splitext(os.path.splitext(os.path.basename(targz_file))[0])[0])
        # open file
        file = tarfile.open(targz_file)
        # read videos
        video_files = [v for v in sorted(file.getnames()) if os.path.splitext(v)[-1] == '.mp4']
        for v in tqdm(video_files):
            # extract mp4 video
            file.extract(v, out_dir)
            # Add to dataset
            try:
                frames = read_video(os.path.join(out_dir, os.path.basename(v)), image_size)
                h5_maker.add_data((frames, t), dtype='uint8')
                if not test:
                    append_to_dict_list(targets, target, count)
                    count += 1
            except StopIteration:
                break
            except (KeyboardInterrupt, SystemExit):
                print("Ctrl+C!!")
                break
            except:
                e = sys.exc_info()[0]
                print("ERROR:", e)
            # Remove mp4 video
            os.remove(os.path.join(out_dir, os.path.basename(v)))
        # close tar
        file.close()
    # close h5
    h5_maker.close()

    # Save targets
    if not test:
        print("\nWriting", os.path.join(out_dir, 'targets.pkl'))
        with open(os.path.join(out_dir, 'targets.pkl'), 'wb') as f:
            pickle.dump(targets, f)

        # Save target_names
        print("Writing", os.path.join(out_dir, 'target_names.txt\n'))
        with open(os.path.join(out_dir, 'target_names.txt'), 'w') as f:
            f.writelines(target_names)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--k600_targz_dir', type=str, help="Path to 'k600_targz' ")
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--out_dir2', type=str, default=None, help="Directory to save .hdf5 files")
    parser.add_argument('--image_size2', type=int, default=None)
    parser.add_argument('--videos_per_class', type=int, default=None)
    parser.add_argument('--video_idx_per_class', type=int, default=0)
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_kinetics_multi2(out_dir=os.path.join(args.out_dir, "train"), kinetics600_dir=os.path.join(args.k600_targz_dir, "train"), image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5,
                                 videos_per_class=args.videos_per_class, video_idx_per_class=args.video_idx_per_class, out_dir2=os.path.join(args.out_dir2, "train"), image_size2=args.image_size2)
    make_h5_from_kinetics_multi2(out_dir=os.path.join(args.out_dir, "val"), kinetics600_dir=os.path.join(args.k600_targz_dir, "val"), image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5,
                                 videos_per_class=args.videos_per_class, video_idx_per_class=args.video_idx_per_class, out_dir2=os.path.join(args.out_dir2, "val"), image_size2=args.image_size2)
    make_h5_from_kinetics_multi2(out_dir=os.path.join(args.out_dir, "test"), kinetics600_dir=os.path.join(args.k600_targz_dir, "test"), image_size=args.image_size, test=True, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5,
                                 videos_per_class=args.videos_per_class, video_idx_per_class=args.video_idx_per_class, out_dir2=os.path.join(args.out_dir2, "test"), image_size2=args.image_size2)
