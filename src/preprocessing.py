import torch
from skimage.color import rgb2lab


class ToLAB(object):
    def __call__(self, image):
        lab_image = rgb2lab(image)
        L = lab_image[:, :, 0:1] / 100
        AB = lab_image[:, :, 1:] / 128
        return L, AB


class ReshapeChannelFirst(object):
    def __call__(self, LAB):
        L, AB = LAB
        return L.transpose(2, 0, 1), AB.transpose(2, 0, 1)


class ToTensor(object):
    def __call__(self, LAB):
        L, AB = LAB
        return torch.from_numpy(L).float(), torch.from_numpy(AB).float()
