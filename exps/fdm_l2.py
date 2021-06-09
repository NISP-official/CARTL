from src.cli.utils import get_model, get_test_dataset, get_train_dataset
import torch

from src import settings
from src.utils import logger
from src.trainer import RobustPlusSingularRegularizationTrainer

from src.attack import LinfPGDAttack
from src.networks import wrn34_10
from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader


def cartl(model, num_classes, dataset, random_init, epsilon, step_size, num_steps, k, lambda_):
    """Cooperative Adversarially-Robust TransferLearning"""
    save_name = f"cartl_{model}_{dataset}_{k}_{lambda_}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    params = {
        "random_init": random_init,
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "dataset_name": dataset
    }
    trainer = RobustPlusSingularRegularizationTrainer(
        k=k,
        _lambda=lambda_,
        model=get_model(model, num_classes),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        attacker=LinfPGDAttack,
        params=params,
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth"
    )
    trainer.train(f"{settings.model_dir / save_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--epsilon", type=float, default=8/255)
    parser.add_argument("--step-size", type=float, default=2/255)
    parser.add_argument("--num-steps", type=int, default=7)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-l", "--lambda_", type=float, required=True)
    parser.add_argument("--random-init", action="store_false") #default value is True

    args = parser.parse_args()

    cartl(
        model=args.model, 
        num_classes=args.num_classes, 
        dataset=args.dataset, 
        random_init=args.random_init, 
        epsilon=args.epsilon, 
        step_size=args.step_size, 
        num_steps=args.num_steps,
        k=args.k,
        lambda_=args.lambda_, 
    )