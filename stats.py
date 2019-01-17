import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import ImageDataset
from preprocessing import ToLAB, ReshapeChannelFirst, ToTensor
from torchvision import transforms
from model import models

def stats(
  images_path,
  model,
  device,
  load,
  log_output
):
  colorizer = models[model]().to(device)
  colorizer.eval()
  colorizer.load_state_dict(torch.load(load, map_location=device))

  image_dataset = ImageDataset(
    images_path,
    transform=transforms.Compose([
      ToLAB(), 
      ReshapeChannelFirst(),
      ToTensor()
    ])
  )

  criterion = nn.MSELoss(size_average=False)

  images = DataLoader(image_dataset, batch_size=1)

  if log_output != None:
    with open(log_output, "w+") as log_file:
      log_file.write("i,loss\n")


  losses = []
  for i, (L, AB) in enumerate(images):
    L = L.to(device)
    AB_pred = colorizer(L).cpu()
    AB = AB.cpu()
    loss = criterion(AB_pred, AB)
    losses.append(loss.item())
    if log_output != None:
      print("{},{}".format(i, loss.item()))
      with open(log_output, "a+") as log_file:
        log_file.write("{},{}\n".format(i, loss.item()))
    
  return np.array(losses)
  #losses = np.array(losses)
  #mean = losses.mean()
  #var = losses.var()
  #z = 1.96
  #lower = mean - z * (var / np.sqrt(len(losses)))
  #upper = mean + z * (var / np.sqrt(len(losses)))
  #print("mean: {}\nvar: {}\nlower: {}\nupper: {}".format(mean, var, lower, upper))