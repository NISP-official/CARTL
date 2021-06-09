# CARTL: Cooperative Adversarially-Robust Transfer Learning 

Code for [ICML'21 paper](https://arxiv.org/abs/2106.06667), *CARTL: Cooperative Adversarially-Robust Transfer Learning*.

### Prerequisites

- python==3.7
- pytorch==1.5.1
- torchvision==0.6.1
- tensorboard==2.4.0
- tensorboard-plugin-wit==1.7.0
- toml==0.10.2
- matplotlib==3.2.2
- pydantic==1.5.1
- pydantic[dotenv]
- pandas==1.2.0
- foolbox==3.2.1

You can download these packages manually or run `pip install -r requirements.txt`

For command-line interface support, run `pip install --editable .`

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Code Overview

```
.
├── exps            # Code for experiments 
├── scripts         # Script for simplifying experiments
├── src 			
│   ├── cli         # Code for command-line interface
│   ├── networks    # Code for defining network architectures
│   ├── trainer     # Code for training processes
│   └── utils       # Code for utilities(logging, dataloader, etc.)
└── tests           # Code for testing key components
```

Please run the script `./scripts/init.sh`  for creating required directories `checkpoint`, `trained_models`, `misc_results`(containing experimental results), `logs` (containing running logs) and `runs`(containing files for tensorboard).

### Trained Models

Trained models are available on [MEGA](https://mega.nz/folder/J5YC3BoD#8uwosiSA5zDhsA0OI_p5EQ).

### Simple Instructions

#### Train an adversarial teacher model

```bash
python -m exps.adversarial_training -m <MODEL_ARCH> -n <NUM_CLASSES> -d <DATASET>
```

Or using cli tools (run `cli at --help` for details)

#### Train a fdm teacher model

```bash
python -m exps.fdm_l2 -m <MODEL_ARCH> -n <NUM_CLASSES> -d <DATASET> -k <K> -l <LAMBDA>
```

Or using cli tools (run `cli fdm --help` for details)

#### Transfer learning

```bash
python -m exps.transfer_learning -m <MODEL_ARCH> -n <NUM_CLASSES> -d <DATASET> -k <K> -t <TEACHER> \
                [--freeze-bn --reuse-teacher-statistic --reuse-statistic]
```

Or using cli tools (run `cli tl --help` for details)

#### Transfer learning with NEFT

```bash
python -m exps.neft_spectrum_norm -m <MODEL_ARCH> -n <NUM_CLASSES> -d <DATASET> -k <K> -t <TEACHER> \
                --power-iter <{POWER_ITER> \
                --norm-beta <NORM_BETA> \
                [--freeze-bn --reuse-teacher-statistic --reuse-statistic]
```

Or change parameters in `./script/pwrn34_neft_sn_norm.sh` and run

```
./script/pwrn34_neft_sn_norm.sh
```

Or using cli tools (run `cli sntl --help` for details)

#### Lwf

```bash
cli lwf -m <MODEL_ARCH> -n <NUM_CLASSES> -d <DATASET> -l <LAMBDA> -t <TEACHER>
```

### Reference

```
@inproceedings{chen2021cartl,
 author = {Chen, Dian and Hu, Hongxin and Wang, Qian and Li, Yinli and Wang, Cong and Shen, Chao and Li, Qi},
 title = {CARTL: Cooperative Adversarially-Robust Transfer Learning},
 BOOKTITLE = {International Conference on Machine Learning},
 YEAR = {2021}
}
```

