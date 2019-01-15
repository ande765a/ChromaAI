import numpy as np
import math
import torch.nn as nn

def calculateStats(input_images, output_images, N, z):
    loss_func = nn.MSELoss(size_average=False)
    image_errors = np.zeros(N, dtype=float)

    for i in enumerate(input_images):
        image_errors = loss_func(output_images[i][:, :, 1:], input_images[i][:, :, 1:])

    error_sums = np.sum(image_errors)
    mean_error = error_sums / N   
    std_dev = np.sum(image_errors - mean_error) / (N - 1)

    lower_bound = mean_error - z * math.sqrt(std_dev / N)
    upper_bound = mean_error + z * math.sqrt(std_dev / N)

    return mean_error, std_dev, lower_bound, upper_bound