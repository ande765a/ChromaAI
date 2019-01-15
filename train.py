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
from model import models

from torchvision import transforms
from torch.utils.data import DataLoader


def train(
        images_path,
        device,
        model,
        load,
        batch_size,
        shuffle,
        num_workers,
        num_epochs,
        learning_rate,
        save_best,
        save,
        save_frequency  # Save after 10 epochs
):

    phases = ["validation", "training"]

    images_dataset = {
        x: ImageDataset(
            os.path.join(images_path, x),
            transform=transforms.Compose(
                [ToLAB(), ReshapeChannelFirst(),
                 ToTensor()]))
        for x in phases
    }

    images = {
        x: DataLoader(
            images_dataset[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        for x in phases
    }

    # Make instance of model
    colorizer = models[model]()

    if load != None:
        print("Loading model from: {}".format(load))
        colorizer.load_state_dict(torch.load(load))

    colorizer = colorizer.to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(colorizer.parameters(), lr=learning_rate)

    best_validation_loss = math.inf

    # Train our model
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        running_train_loss = 0.0
        running_validation_loss = 0.0

        for phase in phases:
            if phase == "training":
                colorizer.train()
            elif phase == "validation":
                colorizer.eval()

            for L, AB in images[phase]:
                L = L.to(device)
                AB = AB.to(device)
                AB_pred = colorizer(L)
                loss = criterion(AB_pred, AB)

                if phase == "training":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * L.size(0)
                elif phase == "validation":
                    running_validation_loss += loss.item() * L.size(0)


        epoch_train_loss = running_train_loss / len(images_dataset["training"])
        epoch_validation_loss = running_validation_loss / len(images_dataset["validation"])
        print("Training loss: {}".format(epoch_train_loss))
        print("Validation loss: {}".format(epoch_validation_loss))

        if save_best != None and epoch_validation_loss < best_validation_loss:
            print("Saving best model.")
            best_validation_loss = epoch_validation_loss
            torch.save(colorizer.state_dict(), save_best)

        if save != None and epoch % save_frequency == 0:
            print("Saving model.")
            torch.save(colorizer.state_dict(), save)

        print("-" * 30)

    if save != None:
        print("Saving final model.")
        torch.save(colorizer.state_dict(), save)
