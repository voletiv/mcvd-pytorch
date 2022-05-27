import numpy as np
import torch

from torchvision import datasets, transforms


class ToTensor(object):
    """Converts a numpy.ndarray (... x H x W x C) to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, scale=True):
        self.scale = scale
    def __call__(self, arr):
        if isinstance(arr, np.ndarray):
            video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))
            if self.scale:
                return video.float()
            else:
                return video.float()
        else:
            raise NotImplementedError


# https://github.com/edenton/svg/blob/master/data/moving_mnist.py
class StochasticMovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, data_root, train=True, seq_len=20, num_digits=2, image_size=64, deterministic=False,
                 step_length=0.1, total_videos=-1, with_target=False, transform=transforms.Compose([ToTensor()])):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = step_length
        self.with_target = with_target
        self.transform = transform
        self.deterministic = deterministic

        self.seed_is_set = False # multi threaded loading
        self.digit_size = 32
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) if total_videos == -1 else total_videos

        print(f"Dataset length: {self.__len__()}")

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.

        if self.with_target:
            targets = np.array(x >= 0.5, dtype=float)

        if self.transform is not None:
            x = self.transform(x)
            if self.with_target:
                targets = self.transform(targets)

        if self.with_target:
            return x, targets
        else:
            return x
