import torch
import time
import os
import json

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize
from src import settings
from src.config import set_seed
from src.cli.utils import get_model
from src.utils import (logger, get_mean_and_std,
                        get_cifar_test_dataloader,
                        get_mnist_test_dataloader,
                        get_svhn_test_dataloader,
                        get_gtsrb_test_dataloder)

from foolbox.attacks import LinfPGD
from foolbox.attacks.base import Attack
from foolbox import PyTorchModel


EPSILON = 8/255
STEP_SIZE = 2/255

SupportDatasetList = ['cifar10', 'cifar100', 'mnist', 'svhn', 'svhntl', 'gtsrb']

def make_eps(dataset: str) -> None:
    global EPSILON, STEP_SIZE

    if dataset == "mnist":
        EPSILON = 0.15
        STEP_SIZE = 0.01
    else:
        EPSILON = 8/255
        STEP_SIZE = 2/255

    logger.info(f"using epsion: {EPSILON}, step size: {STEP_SIZE}")

    
def get_test_dataset(dataset: str, batch_size=256, normalize=False) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        return get_cifar_test_dataloader(dataset=dataset, normalize=normalize, shuffle=False, batch_size=batch_size)
    elif dataset == 'mnist':
        return get_mnist_test_dataloader(normalize=normalize, shuffle=False, batch_size=batch_size)
    elif dataset.startswith('svhn'):
        # 'svhn': using mean and std of 'svhn'
        # 'svhn': using mean and std of 'cifar100'
        return get_svhn_test_dataloader(dataset_norm_type=dataset, normalize=normalize, shuffle=False, batch_size=batch_size)
    elif dataset == "gtsrb":
        return get_gtsrb_test_dataloder(normalize=False, batch_size=batch_size)

def get_attacker(attacker:str="LinfPGD")->Attack:
    import re
    if attacker.startswith("LinfPGD"):
        if attacker == "LinfPGD":
            _steps = 100
        else:
            if re.match(r"LinfPGD-(\d+)", attacker) is None:
                raise ValueError("using 'LinfPGD' or 'LinfPGD-X'")
            [_, _steps, _] = re.split(r"LinfPGD-(\d+)", attacker)
        logger.info(f"using attack: PGD-{_steps}")
        return LinfPGD(steps=int(_steps), abs_stepsize=STEP_SIZE)
    else:
        raise ValueError(f"not support attacker type '{attacker}'")

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("_mean", torch.tensor(mean).view(3, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(3, 1, 1))
    
    def forward(self, x):
        x = (x - self._mean) / self._std
        return self._model(x)

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
    
    logger.info(f"Accuracy: {acc/items}")
    return acc/items


def freeze_model_trainable_params(model:torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    logger.debug("all parameters are freezed")


def robust(model, attacker:Attack, testset, device, total_size=None):
    fmodel = PyTorchModel(model, bounds=(0, 1), device=device)

    items = 0
    rob = 0

    for it, (data, labels) in enumerate(testset):
        data = data.to(device)
        labels = labels.to(device)

        raw_advs, clipped_advs, success = attacker(fmodel, data, labels, epsilons=EPSILON)

        rob += (labels.shape[0] - success.sum().item())
        items += labels.shape[0]

        if total_size is not None and items > total_size:
            break
    
    logger.info(f"Robust: {rob/items}%")

    return rob / items

def exp(model_path, args):
    set_seed(settings.seed)

    testset = get_test_dataset(args.dataset, normalize=False, batch_size=128)

    model = get_model(args.model_type, args.num_classes, args.k).to(settings.device)
    model.load_state_dict(torch.load(model_path, map_location=settings.device))
    logger.debug(f"load from `{model_path}`")
    model.eval()
    freeze_model_trainable_params(model)

    mean, std = get_mean_and_std(args.dataset)
    model = NormalizationWrapper(model, mean, std).to(settings.device)
    model.eval()

    acc = accuracy(model, testset, settings.device)

    attacker = get_attacker(args.attacker)
    start_time = time.perf_counter()
    rob = robust(model, attacker, testset, settings.device, total_size=args.total_size)
    end_time = time.perf_counter()
    logger.info(f"costing time: {end_time-start_time:.2f} secs")

    result = {
        model_path: {
                "Acc": acc,
                f"{args.attacker} Rob": rob
            }
        }

    logger.info(result)
    if args.result_file is not None:
        if not os.path.exists(os.path.dirname(args.result_file)):
            os.makedirs(os.path.dirname(args.result_file))
        
        if os.path.exists(args.result_file):
            with open(args.result_file, "r") as f:
                exist_data = json.load(f)
            for key in result.keys():
                if key in exist_data:
                    for elem in result[key].keys():
                        exist_data[key][elem] = result[key][elem]
                else:
                    exist_data[key] = result[key]
            result = exist_data
        
        with open(args.result_file, "w+") as f:
            json.dump(result, f)

if __name__ == "__main__":
    import argparse

    parser  = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-n", "--num_classes", type=int, required=True)
    parser.add_argument("-a", "--attacker", type=str, default="LinfPGD")
    parser.add_argument("-k", "--k", type=int, default=1)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--log", type=str, default="attack.log")
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--total-size", type=int, default=None)
    args = parser.parse_args()
    logger.change_log_file(settings.log_dir / args.log)

    make_eps(args.dataset)

    if args.model is None:
        model_list = [

        ]
    else:
        model_list = [args.model] 
    
    
    for model_path in model_list:
        exp(model_path, args)
