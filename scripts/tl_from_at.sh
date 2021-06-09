#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models
MODEL_ARCH=wrn34
SOURCE_DOMAIN=cifar100
TARGET_DOMAIN=cifar10
NUM_CLASSES=10

FREEZE_BN="--freeze-bn"
REUSE_STATISTIC="" #"--reuse-statistic"
REUSE_TEACHER_STATISTIC="--reuse-teacher-statistic"

TERM=`python -m exps.utils ${REUSE_STATISTIC} ${REUSE_TEACHER_STATISTIC} ${FREEZE_BN}`
valid $?

RESULT_FILE=misc_results
FOOLBOX_LOG_FILE_NAME=${SOURCE_DOMAIN}_${TARGET_DOMAIN}_foolbox

for k in 4 6 8; do
    TEACHER_MODEL="at_${MODEL_ARCH}_${SOURCE_DOMAIN}-best_robust"

    echo "#######################################################"
    echo "simply transferring from ${TEACHER_MODEL}"
    echo "output tl_${TERM}_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL}-last"

    python -m exps.transfer_learning --model=${MODEL_ARCH} --num_classes=${NUM_CLASSES} --dataset=${TARGET_DOMAIN} --teacher=${TEACHER_MODEL} -k=${k} ${REUSE_TEACHER_STATISTIC} ${FREEZE_BN} ${REUSE_STATISTIC}
    valid $?

    python -m exps.foolbox_bench -d=${TARGET_DOMAIN} -n=${NUM_CLASSES} --model-type=${MODEL_ARCH} -m=${MODEL_PATH}/tl_${TERM}_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL}-last \
                -k=${k} --log=${FOOLBOX_LOG_FILE_NAME}.log --result-file=${RESULT_FILE}/${FOOLBOX_LOG_FILE_NAME}.json --attacker=LinfPGD-100
    valid $?

done



