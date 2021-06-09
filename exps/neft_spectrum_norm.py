from typing import Optional, ValuesView

from torch.nn import Module
from torch.utils.data import DataLoader

import torch

from src import settings

from src.utils import logger

from src.trainer import SpectralNormTransferLearningTrainer

from src.attack import LinfPGDAttack

from src.cli.utils import get_train_dataset, get_test_dataset
from src.cli.utils import get_model

def sn_tl(model, num_classes, dataset, k, teacher, power_iter, norm_beta, freeze_bn, reuse_statistic, reuse_teacher_statistic):
    """transform leanring"""
    from .utils import make_term
    term = make_term(freeze_bn, reuse_statistic, reuse_teacher_statistic)
    save_name = f"sntl_{power_iter}_{norm_beta}_{term}_{model}_{dataset}_{k}_{teacher}_{settings.seed}"

    logger.change_log_file(f"{settings.log_dir / save_name}.log")

    trainer = SpectralNormTransferLearningTrainer(
        k=k,
        teacher_model_path=str(settings.model_dir / teacher),\
        model=get_model(model=model, num_classes=num_classes, k=k),
        train_loader=get_train_dataset(dataset=dataset),
        test_loader=get_test_dataset(dataset=dataset),
        checkpoint_path=f"{settings.checkpoint_dir / save_name}.pth",
        power_iter=power_iter,
        norm_beta=norm_beta,
        freeze_bn=freeze_bn,
        reuse_statistic=reuse_statistic,
        reuse_teacher_statistic=reuse_teacher_statistic,
    )

    trainer.train(f"{settings.model_dir / save_name}")

if __name__ == '__main__':
    import argparse
    class FreezeBNAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                freeze_bn = True
            else:
                if len(values) == 1:
                    import re
                    if re.match(r"\((\d+),(\d+)\)", values[0]) is None:
                        raise ValueError("Assume you provide a tuple. please pass it in '(min, max)'.")
                    [_, min_, max_, _] = re.split(r"\((\d+),(\d+)\)", values[0])
                    freeze_bn = (int(min_), int(max_))
                else:
                    freeze_bn = list(map(lambda x: int(x), values))

            setattr(namespace, self.dest, freeze_bn)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-t", "--teacher", type=str)
    parser.add_argument("--power-iter", type=int, default=1)
    parser.add_argument("--norm-beta", type=float, default=1.0)
    parser.add_argument("--freeze-bn", action=FreezeBNAction, nargs="*", default=False)
    parser.add_argument("--reuse-statistic", action="store_true")
    parser.add_argument("--reuse-teacher-statistic", action="store_true")

    args = parser.parse_args()
    # print(args)

    sn_tl(model=args.model,
        num_classes=args.num_classes,
        dataset=args.dataset,
        k=args.k,
        teacher=args.teacher,
        power_iter=args.power_iter,
        norm_beta=args.norm_beta,
        freeze_bn=args.freeze_bn,
        reuse_statistic=args.reuse_statistic,
        reuse_teacher_statistic=args.reuse_teacher_statistic
    )
