# https://github.com/edenton/svg/blob/master/data/convert_bair.py
import argparse
import cv2
import glob
import numpy as np
import os
import pickle
import sys

from tqdm import tqdm

from h5 import HDF5Maker


class KTH_HDF5Maker(HDF5Maker):

    def add_video_info(self):
        pass

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('person')
        self.writer.create_group('target')

    def add_video_data(self, data, dtype=None):
        data, person, target = data
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['person'].create_dataset(str(self.count), data=person, dtype='uint8')
        self.writer['target'].create_dataset(str(self.count), data=target, dtype='uint8')
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")


def read_video(video, image_size):

    cap = cv2.VideoCapture(video)
    frames = []

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            # print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray, (image_size, image_size))
        frames.append(image)

    cap.release()
    return frames


def show_video(frames):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    im1 = plt.imshow(frames[0])
    def update(frame):
        im1.set_data(frame)
    ani = FuncAnimation(plt.gcf(), update, frames=frames, interval=10, repeat=False)
    plt.show()


def append_to_dict_list(d, key, value):
    if key not in d:
        d[key] = []
    d[key].append(value)


def make_h5_from_kth(kth_dir, image_size=64, out_dir='./h5_ds', vids_per_shard=1000000, force_h5=False):

    # H5 maker
    h5_maker = KTH_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    # data_root = '/path/to/Datasets/KTH'
    # image_size = 64
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    # frame_rate = 25

    count = 0
    persons = {}
    targets = {}
    for video in tqdm(sorted(glob.glob(os.path.join(kth_dir, 'raw', '*', '*')))):

        try:
            frames = read_video(video, image_size)
            person = int(os.path.basename(video).split('_')[0].split('person')[-1])
            target = classes.index(video.split('/')[-2])
            h5_maker.add_data((frames, person, target), dtype='uint8')
            append_to_dict_list(persons, person, count)
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

    h5_maker.close()

    # Save persons
    print("Writing", os.path.join(out_dir, 'persons.pkl'))
    with open(os.path.join(out_dir, 'persons.pkl'), 'wb') as f:
        pickle.dump(persons, f)

    # Save targets
    print("Writing", os.path.join(out_dir, 'targets.pkl'))
    with open(os.path.join(out_dir, 'targets.pkl'), 'wb') as f:
        pickle.dump(targets, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--kth_dir', type=str, help="Directory with KTH")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vids_per_shard', type=int, default=1000000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_kth(out_dir=args.out_dir, kth_dir=args.kth_dir, image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
