# https://github.com/podgorskiy/StyleGan/blob/master/dataloader.py

# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# import dareblopy as db
import numpy as np
import torch


class TFRecordsDataLoader:
    def __init__(self, tfrecords_paths, batch_size,
                 ch=3, img_size=None, length=None, seed=0, buffer_size_mb=200):
        self.iterator = None
        self.filenames = tfrecords_paths
        self.batch_size = batch_size
        self.ch = ch
        self.img_size = img_size
        self.length = length
        self.seed = seed
        self.buffer_size_mb = buffer_size_mb

        if self.img_size is None or self.ch is None:
            raw_dataset = tf.data.TFRecordDataset(self.filenames[0])
            for raw_record in raw_dataset.take(1): pass
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            # print(example)
            result = {}
            # example.features.feature is the dictionary
            for key, feature in example.features.feature.items():
              # The values are the Feature objects which contain a `kind` which contains:
              # one of three fields: bytes_list, float_list, int64_list
              kind = feature.WhichOneof('kind')
              result[key] = np.array(getattr(feature, kind).value)
            # ch, img_size
            self.ch = result['shape'][0]
            self.img_size = result['shape'][-1]

        if self.length is None:
            import tensorflow as tf
            tf.compat.v1.enable_eager_execution()
            self.length = 0
            for file in self.filenames:
                self.length += sum(1 for _ in tf.data.TFRecordDataset(file))

        self.features = {
            # 'shape': db.FixedLenFeature([3], db.int64),
            'data': db.FixedLenFeature([ch, img_size, img_size], db.uint8)
        }

        self.buffer_size = 1024 ** 2 * self.buffer_size_mb // (3 * img_size * img_size)

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.filenames, self.features, self.batch_size, self.buffer_size, seed=self.seed)

    def transform(self, x):
        return torch.from_numpy(x[0]), torch.zeros(len(x[0]))

    def __iter__(self):
        return map(self.transform, self.iterator)

    def __len__(self):
        return self.length // self.batch_size


class FFHQ_TFRecordsDataLoader(TFRecordsDataLoader):
    def __init__(self, tfrecords_paths, batch_size, img_size,
                 seed=0, length=70000, buffer_size_mb=200):
        super().__init__(tfrecords_paths, batch_size, img_size=img_size, seed=seed,
                         length=length, buffer_size_mb=buffer_size_mb)
