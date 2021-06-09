from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import NormalTrainer

from src.cli.utils import get_train_dataset, get_test_dataset
from src.cli.utils import get_model

def nt(model, num_classes, dataset):
    save_name = f"nt_{model}_{dataset}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = NormalTrainer(
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)

    args = parser.parse_args()

    nt(model=args.model, num_classes=args.num_classes, dataset=args.dataset)