# https://github.com/edenton/svg/blob/master/data/convert_bair.py
import argparse
import glob
import imageio
import io
import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image
from tensorflow.python.platform import gfile
from tqdm import tqdm

from h5 import HDF5Maker


def get_seq(data_dir, dname):
    data_dir = '%s/softmotion30_44k/%s' % (data_dir, dname)

    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k = 0
        # tf.enable_eager_execution()
        for serialized_example in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                # image_seq.append(byte_str)
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                image_seq.append(arr)
            # image_seq = np.concatenate(image_seq, axis=0)
            k = k + 1
            yield f, k, image_seq


def make_h5_from_bair(bair_dir, split='train', out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    seq_generator = get_seq(bair_dir, split)

    filenames = gfile.Glob(os.path.join('%s/softmotion30_44k/%s' % (bair_dir, split), '*'))
    for file in tqdm(filenames):

        # num = sum(1 for _ in tf.python_io.tf_record_iterator(file))
        num = 256
        for i in tqdm(range(num)):

            try:
                f, k, seq = next(seq_generator)
                # h5_maker.add_data(seq, dtype=None)
                h5_maker.add_data(seq, dtype='uint8')

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
    parser.add_argument('--bair_dir', type=str, help="Directory with videos")
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_bair(out_dir=os.path.join(args.out_dir, 'train'), bair_dir=args.bair_dir, split='train', vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
    make_h5_from_bair(out_dir=os.path.join(args.out_dir, 'test'), bair_dir=args.bair_dir, split='test', vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
