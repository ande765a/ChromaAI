import os
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.image_filenames = os.listdir(images_path)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_path, image_filename)
        image = 1.0 / 255 * np.array(img_to_array(load_img(image_path)))

        if self.transform:
            return self.transform(image)

        return image
