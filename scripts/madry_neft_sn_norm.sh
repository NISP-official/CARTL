#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models
POWER_ITER=1
NORM_BETA=0.6
FREEZE_BN_LAYER=True


for k in 4; do
    TEACHER_MODEL=at_wrn34_cifar100-best_robust

    if [ ${FREEZE_BN_LAYER} = "True" ]; then
        freeze_bn_op="--freeze-bn"
    else
        freeze_bn_op=""
    fi

    echo "#######################################################"
    echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
    python -m exps.neft_spectrum_norm --model=pwrn34 --num_classes=10 --dataset=cifar10 --teacher=${TEACHER_MODEL} -k=${k} \
             --power-iter=${POWER_ITER} --norm-beta=${NORM_BETA} ${freeze_bn_op}
    valid $?

    python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_${FREEZE_BN_LAYER}_pwrn34_cifar10_${k}_${TEACHER_MODEL}-last -k=${k} \
            --log=sntl.log --result-file=logs/sntl.json --model-type=pwrn34 --dataset=cifar10 --num_classes=10
    valid $?

done
