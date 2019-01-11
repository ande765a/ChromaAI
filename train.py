# -*- coding: utf-8 -*-
"""# Import dependencies"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import ToLAB, ReshapeChannelFirst, ToTensor
from data import ImageDataset
from model import Colorizer

from torchvision import transforms
from torch.utils.data import DataLoader


def train(
        images_path,
        device,
        load,
        batch_size,
        shuffle,
        num_workers,
        num_epochs,
        learning_rate,
        save,
        save_frequency  # Save after 10 epochs
):

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

    if load != None:
        print("Loading model from: {}".format(load))
        colorizer.load_state_dict(torch.load(load))

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(colorizer.parameters(), lr=learning_rate)

    # Train our model
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        running_train_loss = 0.0
        running_test_loss = 0.0

        for phase in ["train", "test"]:

            if phase == "train":
                colorizer.train()
            elif phase == "test":
                colorizer.eval()

            for L, AB in images[phase]:
                L = L.to(device)
                AB = AB.to(device)
                AB_pred = colorizer(L)
                loss = criterion(AB_pred, AB)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * L.size(0)
                elif phase == "test":
                    running_test_loss += loss.item() * L.size(0)

        epoch_train_loss = running_train_loss / len(images_dataset)
        epoch_test_loss = running_test_loss / len(images_dataset)
        print("Train loss: {}".format(epoch_train_loss))
        print("Test loss: {}".format(epoch_test_loss))

        if save != None and epoch % save_frequency == 0:
            print("Saving model.")
            torch.save(colorizer.state_dict(), save)

        print("-" * 30)


