#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models
POWER_ITER=1
NORM_BETA=0.4

ARCH=pwrn34
TARGET_DOMAIN=cifar10
NUM_CLASSES=10

FREEZE_BN="--freeze-bn"
REUSE_TEACHER_STATISTIC="" #"--reuse-teacher-statistic"
REUSE_STATISTIC=""

TERM=`python -m exps.utils ${FREEZE_BN} ${REUSE_STATISTIC} ${REUSE_TEACHER_STATISTIC}`
valid $?


for k in 4 6 8; do
    TEACHER_MODEL=cartl_wrn34_cifar100_${k}_0.005-best_robust

    if [ ! -e "${MODEL_PATH}/${TEACHER_MODEL}" ]; then
        echo "CANNOT FIND ${MODEL_PATH}/${TEACHER_MODEL}, SKIPPING"
        continue
    fi

    STUDENT_MODEL=sntl_${POWER_ITER}_${NORM_BETA}_${TERM}_${ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL}-last

    echo "#######################################################"
    echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
    echo "resulting in ${STUDENT_MODEL}"
    
    python -m exps.neft_spectrum_norm --model=${ARCH} --teacher=${TEACHER_MODEL} \
                --k=${k} \
                --num_classes=${NUM_CLASSES} \
                --dataset=${TARGET_DOMAIN} \
                --power-iter=${POWER_ITER} \
                --norm-beta=${NORM_BETA} \
                ${FREEZE_BN} ${REUSE_STATISTIC} ${REUSE_TEACHER_STATISTIC}

    valid $?


    python -m exps.foolbox_bench -d=${TARGET_DOMAIN} -n=${NUM_CLASSES} --model-type=${ARCH} -m=${MODEL_PATH}/${STUDENT_MODEL} \
            -k=${k} --log=foolbox_${k}.log --result-file=misc_results/foolbox_${k}.json --attacker=LinfPGD-100
    valid $?

done
