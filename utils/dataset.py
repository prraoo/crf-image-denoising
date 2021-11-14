from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, dir_noisy_image, dir_true_image, scale=1, gt_suffix=''):
        self.dir_noisy_image = dir_noisy_image
        self.dir_true_image = dir_true_image
        self.scale = scale
        self.gt_suffix = gt_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(dir_noisy_image)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        gt_file = glob(self.dir_true_image + idx + self.gt_suffix + '.*')
        img_file = glob(self.dir_noisy_image + idx + '.*')

        assert len(gt_file) == 1, \
            f'Either no gt or multiple gts found for the ID {idx}: {gt_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        gt = Image.open(gt_file[0])
        img = Image.open(img_file[0])

        assert img.size == gt.size, \
            f'Image and gt {idx} should be the same size, but are {img.size} and {gt.size}'

        img = self.preprocess(img, self.scale)
        gt = self.preprocess(gt, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'gt': torch.from_numpy(gt).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, dir_noisy_image, dir_true_image, scale=1):
        super().__init__(dir_noisy_image, dir_true_image, scale, gt_suffix='_gt')


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)
