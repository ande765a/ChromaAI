import os
import torch
import argparse
from train import train
from eval import eval

parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, help="eval or train", default="eval")
parser.add_argument("--output", type=str, help="Output images")
parser.add_argument("--images-path", type=str, help="")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save", type=str, default=None)
parser.add_argument("--save-frequency", type=int, default=10)
parser.add_argument(
    "--device", type=str, help="What device to use. Ex. 'cuda' or 'cpu'")
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--shuffle", type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == "eval":
        eval(args.images_path, load=args.load, output=args.output, batch_size=args.batch_size)

    elif args.mode == "train":
        device = torch.device(args.device) if args.device else torch.device("cpu")

        train(
            args.images_path,
            device=device,
            load=args.load,
            save=args.save,
            save_frequency=args.save_frequency,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            shuffle=args.shuffle)
    else:
        print("Unknown mode")
