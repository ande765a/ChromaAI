# -*- coding: utf-8 -*-
"""# Import dependencies"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .preprocessing import ToLAB, ReshapeChannelFirst, ToTensor
from .data import ImageDataset
from .model import Colorizer

from torchvision import transforms
from torch.utils.data import DataLoader
#from keras.preprocessing.image import array_to_img
from skimage.color import rgb2lab
from skimage.io import imsave


def train(images_path,
          device,
          batch_size=32,
          shuffle=True,
          num_workers=4,
          num_epochs=10,
          learning_rate=1e-3):

    images_dataset = {
        x: ImageDataset(
            os.path.join(images_path, x),
            transform=transforms.Compose(
                [ToLAB(), ReshapeChannelFirst(),
                 ToTensor()]))
        for x in ["train", "test"]
    }

    images = {
        x: DataLoader(
            images_dataset[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        for x in ["train", "test"]
    }

    # Make instance of model
    colorizer = Colorizer().to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(colorizer.parameters(), lr=learning_rate)

    # Train our model
    loss_history = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        running_loss = 0.0

        for phase in ["train"]:

            for L, AB in images[phase]:
                L = L.to(device)
                AB = AB.to(device)

                optimizer.zero_grad()
                AB_pred = colorizer(L)
                loss = criterion(AB_pred, AB)
                #loss_history.append(loss.item())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * L.size(0)

        epoch_loss = running_loss / len(images_dataset)
        loss_history.append(epoch_loss)
        print("Loss: {}".format(epoch_loss))
        print("-" * 30)

    #torch.save(colorizer.state_dict(), "/content/drive/My Drive/model-2019-01-10-1100")


#with torch.no_grad():
#    L, AB = next(iter(images["test"]))
#    L = L.to(device)
#    AB_pred = colorizer(L).cpu()
#    L = L.cpu()
#    output_images = torch.cat((L * 100, AB_pred * 128), dim=1).double()
#    output_images = output_images.numpy().transpose((0, 2, 3, 1))
#    for array_image in output_images:
#        output_image = array_to_img(lab2rgb(array_image))
#        display(output_image)
