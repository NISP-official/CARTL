from typing import Any, Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn

from src import settings
from src.config import set_seed
from src.utils import logger, get_mean_and_std, clamp, evaluate_accuracy

class LinfPGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init: int = 1, epsilon=8/255, step_size=2/255, num_steps=20,
                 loss_function: Callable[[Any], Tensor] = nn.CrossEntropyLoss(),
                 dataset_name: str = settings.dataset_name, device: str = settings.device
                 ):
        dataset_mean, dataset_std = get_mean_and_std(dataset_name)
        mean = torch.tensor(dataset_mean).view(3, 1, 1).to(device)
        std = torch.tensor(dataset_std).view(3, 1, 1).to(device)

        clip_max = ((clip_max - mean) / std)
        clip_min = ((clip_min - mean) / std)
        epsilon = epsilon / std
        step_size = step_size / std

        self.min = clip_min
        self.max = clip_max
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.random_init = random_init
        self.num_steps = num_steps
        self.loss_function = loss_function

    def random_delta(self, delta: Tensor) -> Tensor:
        delta.uniform_(-1, 1)
        delta = delta * self.epsilon

        return delta

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        delta = torch.zeros_like(x)
        if self.random_init:
            delta = self.random_delta(delta)
        xt = x + delta
        xt.requires_grad = True

        for it in range(self.num_steps):
            y_hat = self.model(xt)
            loss = self.loss_function(y_hat, target)

            self.model.zero_grad()
            loss.backward()

            grad_sign = xt.grad.detach().sign()
            xt.data = xt.detach() + self.step_size * grad_sign
            xt.data = clamp(xt - x, -self.epsilon, self.epsilon) + x
            xt.data = clamp(xt.detach(), self.min, self.max)

            xt.grad.data.zero_()

        return xt

    def print_parameters(self):
        params = {
            "min": self.min,
            "max": self.max,
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "num_steps": self.num_steps,
            "random_init": self.random_init,
        }
        params_str = "\n".join([": ".join(map(str, item)) for item in params.items()])
        logger.info(f"using attack: {type(self).__name__}")
        logger.info(f"attack parameters: \n{params_str}")

def pgd_robust(model, testset, params, device):
    attacker = LinfPGDAttack(model=model, device=device, **params)
    attacker.print_parameters()

    items = 0
    rob = 0
    for data, labels in testset:
        data = data.to(device) #type:torch.Tensor
        labels = labels.to(device) #type:torch.Tensor
        data = attacker.calc_perturbation(data, labels)
        
        with torch.no_grad():
            pred = model(data) #type:torch.Tensor
            items += labels.shape[0]
            rob += pred.argmax(dim=1).eq(labels).sum().item()
        
    logger.info(f"PGD-20 accuracy: {rob/items}%")
    return rob/items
    

def accuracy(model, testset, device):
    items = 0
    acc = 0
    with torch.no_grad():
        for data, labels in testset:
            data = data.to(device) #type:torch.Tensor
            labels = labels.to(device) #type:torch.Tensor
            pred = model(data) #type:torch.Tensor
            items += labels.shape[0]
            acc += pred.argmax(dim=1).eq(labels).sum().item()
    
    logger.info(f"Accuracy: {acc/items}%")
    return acc/items

def freeze_model_trainable_params(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    logger.debug("all parameters are freezed")


if __name__ == '__main__':
    from src.networks import parseval_retrain_wrn34_10, wrn34_10, resnet18
    from src.utils import (get_cifar_test_dataloader, get_cifar_train_dataloader, get_mnist_test_dataloader,
                        get_mnist_test_dataloader_one_channel)
    from src.cli.utils import get_test_dataset, get_model

    import time
    import json 
    import argparse
    import os

    parser  = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-n", "--num_classes", type=int, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("-k", "--k", type=int, default=1)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--result-file", type=str, default=None)
    args = parser.parse_args()

    params = {
        "random_init": 1,
        "epsilon": 8/255,
        "step_size": 2/255,
        "num_steps": 20,
        "dataset_name": args.dataset,
    }
    
    if args.model is None:
        model_list = [

        ]
    else:
        model_list = [args.model]
    
    if args.log is None:
        logger.change_log_file(settings.log_dir / f"pgd20_attack.log")
    else:
        logger.change_log_file(settings.log_dir / args.log)

    test_loader = get_test_dataset(args.dataset)
    model = get_model(model=args.model_type, k=args.k, num_classes=args.num_classes)
    logger.warning(f"YOU ARE USING MODEL {type(model).__name__}")

    result = dict()
    for model_path in model_list:
        set_seed(settings.seed)
        model.load_state_dict(torch.load(model_path, map_location=settings.device))
        logger.debug(f"load from `{model_path}`")

        model.to(settings.device)
        model.eval()
        freeze_model_trainable_params(model)

        acc = accuracy(model, test_loader, settings.device)

        start_time = time.perf_counter()
        rob = pgd_robust(model, test_loader, params, settings.device)
        end_time = time.perf_counter()
        logger.info(f"costing time: {end_time-start_time:.2f} secs")

        result[model_path] = {"Acc":acc, "Rob":rob}

    logger.info(result)

    if args.result_file is not None:
        if not os.path.exists(os.path.dirname(args.result_file)):
            os.makedirs(os.path.dirname(args.result_file))
        
        if os.path.exists(args.result_file):
            with open(args.result_file, "r") as f:
                exist_data = json.load(f)
            for key in result.keys():
                exist_data[key] = result[key]
            result = exist_data
        
        with open(args.result_file, "w+") as f:
            json.dump(result, f)
