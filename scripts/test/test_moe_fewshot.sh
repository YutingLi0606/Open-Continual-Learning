##!/bin/bash

set -v
set -e
set -x

GPU=0,1
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
chooser=(Aircraft_autochooser Caltech101_autochooser CIFAR100_autochooser DTD_autochooser EuroSAT_autochooser Flowers_autochooser Food_autochooser MNIST_autochooser OxfordPet_autochooser StanfordCars_autochooser SUN397_autochooser)
threshold=(600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4 600e-4)
num=22 # experts num

# only need to set your ckpt_path
model_ckpt_path=/data/liyuting/CL/f/ckpt/exp_router11_experts22_1000iters_few_shot

# inference
for ((j = 0; j < 11; j++)); do
  for ((i = 0; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[j]}

    CUDA_VISIBLE_DEVICES=${GPU} python main_moe.py --eval-only \
        --train-mode=adapter \
        --eval-datasets=${dataset_cur} \
        --load ${model_ckpt_path}/${dataset[i]}.pth \
        --load_autochooser ${model_ckpt_path}/${chooser[i]}.pth \
        --data-location /data_ssd/mtil_datasets \
        --ffn_adapt_where AdapterDoubleEncoder \
        --ffn_adapt \
        --apply_moe \
        --task_id 200 \
        --multi_experts \
        --experts_num ${num} \
        --autorouter \
        --threshold=${threshold[i]}
    done
done
