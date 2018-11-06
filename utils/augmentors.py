'''
https://github.com/uoguelph-mlrg/Cutout
'''
import torch
import numpy as np
from numpy.random import normal


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RainbowOut(object):
    """
    My own changes include:
        Probability of cutout
        max/min size, again, probabilistic
        width and height differences
        intensity diferences
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,
                 n_holes=1,
                 width_min=16,
                 height_min=16,
                 width_max=17,
                 height_max=17,
                 mu=(0, 0, 0),
                 std=(0.3, 0.3, 0.3)
                 ):
        self.n_holes = n_holes
        self.width_min = width_min
        self.height_min = height_min
        self.width_max = width_max
        self.height_max = height_max
        self.mu = mu
        self.std = std


    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        c = img.size(0)

        mask = np.zeros((c, h, w), np.float32)
        values = np.zeros((c, h, w), np.float32)
        intensities = [normal(self.mu[0], self.std[0])]
        for i in range(1, len(self.mu)):
            intensities.append(normal(self.mu[i], self.std[i]))



        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            widths = np.random.randint(low=self.width_min, high=self.width_max, size=c)
            heights = np.random.randint(low=self.width_min, high=self.width_max, size=c)

            x_offsets = np.random.randint(low=-int(np.round(np.min(heights)/8)), high=int(np.round(np.min(heights)/8)), size=c-1)
            y_offsets = np.random.randint(low=-int(np.round(np.min(widths)/8)), high=int(np.round(np.min(widths)/8)), size=c-1)

            for ci in range(c):
                x_offset = x_offsets[ci-1] if ci != 0 else 0
                y_offset = y_offsets[ci-1] if ci != 0 else 0
                xc1 = (x + x_offset - widths[ci] // 2)
                xc1 = np.clip(xc1, 0, h)
                xc2 = (x + x_offset + widths[ci] // 2)
                xc2 = np.clip(xc2, 0, h)

                yc1 = (y + y_offset - heights[ci] // 2)
                yc1 = np.clip(yc1, 0, h)
                yc2 = (y + y_offset + heights[ci] // 2)
                yc2 = np.clip(yc2, 0, h)

                values[ci, yc1:yc2, xc1:xc2] = intensities[ci]
                mask[ci, yc1:yc2, xc1:xc2] = 1



        mask = torch.from_numpy(mask)
        values = torch.from_numpy(values)

        img[mask == 1] = values[mask == 1]

        return img


class ChannelSwap(object):
    """
    Swap channels
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,
                 swap_p=(0.1, 0.1, 0.1)
                 ):
        self.swap_p = swap_p


    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with swapped channels
        """

        c = img.size(0)



        tmp_img = img.clone()

        channel_order = np.array([0, 0, 0])  # Set all channels to zero to start off safety while loop
        while np.sum(channel_order) == 0: # only really here for safety sake
            channel_order = np.arange(1, c + 1)

            for channel in channel_order:
                prob = np.random.uniform(0,1)
                if prob < self.swap_p[channel - 1]:
                    channel_order[channel-1] = np.random.choice(np.setdiff1d(np.concatenate((channel_order, [0])), channel))


        for i, channel in enumerate(channel_order):
            if i != channel - 1:
                img[i] = tmp_img[channel-1] if channel else 0



        return img