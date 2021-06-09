#!/bin/bash
source scripts/utils.sh

cuda $1

K=$2
if [ $K -ne 4 -a $K -ne 6 -a $K -ne 8 ]; then
    echo "invalid layer $K"
    exit
else
    echo "considering k=${K}"
fi

if [ ! $3 ]; then
    echo "please specify a model list"
    exit
else
    echo "load list from $3"
fi
MODEL_LIST=`cat $3`

MODEL_PATH=trained_models
ARCH=pwrn34

TARGET_DOMAIN=cifar10
NUM_CLASSES=10

LOG_FILE_NAME=foolbox
RESULT_FILE=misc_results

for model in ${MODEL_LIST[@]}; do
    if [ ! -e ${model} ]; then
        echo "cannot find ${model}"
        continue
    else
        echo "using $model"
    fi
    python -m exps.foolbox_bench -d=${TARGET_DOMAIN} -n=${NUM_CLASSES} --model-type=${ARCH} -m=${model} \
            -k=${K} --log=${LOG_FILE_NAME}_${K}.log --result-file=${RESULT_FILE}/${LOG_FILE_NAME}_${K}.json --attacker=LinfPGD-100
    valid $?
done