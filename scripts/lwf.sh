
###############################################
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


function valid () {
    # check whether previous cmd is correctly executed
    if [ $? -ne 0 ]; then
        exit
    fi
}
###############################################

export seed=$2
TEACHER_MODEL=at_wrn34_cifar100-best_robust_${seed}
dataset=svhntl
test_size=26032
model_arch=wrn34

lams=(0.1 0.01 0.005 0.001)
num_classes=10

echo "#######################################################"
echo "lwf transfer from ${TEACHER_MODEL}"

for lam in ${lams[@]}; do
    echo "using lambda, ${lam}"
    cli lwf -m ${model_arch} -n ${num_classes} -d ${dataset} -l ${lam} -t ${TEACHER_MODEL}
    valid $?

    python -m exps.eval_robust_wrn34 --model=trained_models/lwf_${model_arch}_${dataset}_${lam}-best \
            --log=lwf.log --result-file=logs/lwf.json --model-type=${model_arch} --dataset=${dataset} --num_classes=${num_classes}
    valid $?

    python -m exps.foolbox_bench --model=trained_models/lwf_${model_arch}_${dataset}_${lam}_${TEACHER_MODEL}-best \
            --log=lwf_pgd100_attack.log --result-file=logs/final_lwf_${dataset}_${seed}_pgd100_attack.json --model-type=${model_arch} \
            --dataset=${dataset} --num_classes=${num_classes} --total-size ${test_size}
    valid $?
done
