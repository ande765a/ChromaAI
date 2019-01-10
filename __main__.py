import os
import torch
import argparse
from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--images-path", type=str, help="")
parser.add_argument(
    "--device", type=str, help="What device to use. Ex. 'cuda' or 'cpu'")
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--shuffle", type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cpu")

    train(
        args.images_path,
        device=device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        shuffle=args.shuffle)
