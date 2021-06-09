#!/bin/bash
function cuda() {
    # check CUDA_VISIBLE_DEVICES is set or not
    if [[ -z ${CUDA_VISIBLE_DEVICES} ]]; then
        if [ $1 ]; then
            # if not set, try read args
            export CUDA_VISIBLE_DEVICES=$1
        else
            # else manually set CUDA_VISIBLE_DEVICES being 0
            export CUDA_VISIBLE_DEVICES=0
        fi
    fi
    echo "CUDA:${CUDA_VISIBLE_DEVICES} is available"
}

function valid () {
    # check whether previous cmd is correctly executed
    if [ $1 -ne 0 ]; then
        echo "Error Occurs"
        exit
    fi
}
