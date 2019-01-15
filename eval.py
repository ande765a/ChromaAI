import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ImageDataset
from preprocessing import ToLAB, ReshapeChannelFirst, ToTensor
from torchvision import transforms
from keras_preprocessing.image import array_to_img
from skimage.color import lab2rgb
from skimage.io import imsave
from statistics import calculateStats
from model import models


def eval(images_path, model, load, output, device, batch_size=8):
    colorizer = models[model]().to(device)
    colorizer.load_state_dict(torch.load(load, map_location=device))

    image_dataset = ImageDataset(
        images_path,
        transform=transforms.Compose(
            [ToLAB(), ReshapeChannelFirst(),
             ToTensor()]))

    images = DataLoader(image_dataset, batch_size=batch_size)

    with torch.no_grad():
        L, _ = next(iter(images))
        L = L.to(device)
        AB_pred = colorizer(L).cpu()
        L = L.cpu()
        output_images = torch.cat((L * 100, AB_pred * 128), dim=1).double()
        output_images = output_images.numpy().transpose((0, 2, 3, 1))

        #mean_error, std_dev, lower_bound, upper_bound = calculateStats(images, output_images, batch_size, 1.96)
        
        for i, array_image in enumerate(output_images):
            output_image = lab2rgb(array_image)
            imsave(os.path.join(output, "{}.png".format(i)), output_image)

    #print("Sample mean error: %f" % mean_error)
    #print("Standard deviation: %f" % std_dev)
    #print("Confidence interval for true mean: [%f ; %f]" % (lower_bound, upper_bound))