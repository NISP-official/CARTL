from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import BNTransferLearningTrainer

from src.attack import LinfPGDAttack

from src.cli.utils import get_train_dataset, get_test_dataset
from src.cli.utils import get_model


def tl(model, num_classes, dataset, k, teacher, freeze_bn=False, reuse_teacher_statistic=False, reuse_statistic=False):
    """transform leanring"""
    from .utils import make_term
    if not freeze_bn and not reuse_teacher_statistic:
        save_name = f"tl_{model}_{dataset}_{k}_{teacher}"
    else:
        term = make_term(freeze_bn, reuse_statistic, reuse_teacher_statistic)
        save_name = f"tl_{term}_{model}_{dataset}_{k}_{teacher}"
    logger.change_log_file(f"{settings.log_dir / save_name}.log")
    trainer = BNTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),
        model=get_model(model, num_classes, k),
        train_loader=get_train_dataset(dataset),
        test_loader=get_test_dataset(dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        freeze_bn=freeze_bn,
        reuse_teacher_statistic=reuse_teacher_statistic,
        reuse_statistic=reuse_statistic
    )
    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-t", "--teacher", type=str)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--reuse-teacher-statistic", action="store_true")
    parser.add_argument("--reuse-statistic", action="store_true")
    parser.add_argument("--random-seed", type=int, default=None)

    args = parser.parse_args()

    if args.random_seed is not None:
        from src.config import set_seed
        settings.seed = args.random_seed
        set_seed(settings.seed)

    tl(model=args.model, num_classes=args.num_classes, dataset=args.dataset, k=args.k, teacher=args.teacher,
            freeze_bn=args.freeze_bn, reuse_teacher_statistic=args.reuse_teacher_statistic, reuse_statistic=args.reuse_statistic)
