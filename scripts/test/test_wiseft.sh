#!/bin/bash

set -e

MODEL_CKPT_PATH="${1:-ckpt/exp_wiseft}"
GPU="${2:-0,1,2,3}"
TEST_MODE="${3:-whole}"
datasets=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)


total_tests=$((${#datasets[@]} * ${#datasets[@]}))
current_test=0

echo "🚀 开始矩阵评估 (总共 $total_tests 个测试)..."
echo ""

# 外层循环：遍历每个训练好的模型
for model_dataset in "${datasets[@]}"; do
    model_path="$MODEL_CKPT_PATH/${model_dataset}.pth"
    
    if [ ! -f "$model_path" ]; then
        echo "⚠️  模型文件不存在，跳过: $model_path"
        continue
    fi
    
    echo "🔧 使用模型: $model_dataset ($model_path)"
    echo "Checkpoint loaded from $model_path"
    
    # 内层循环：在每个数据集上评估当前模型
    for test_dataset in "${datasets[@]}"; do
        current_test=$((current_test + 1))
        progress=$(echo "scale=1; $current_test * 100 / $total_tests" | bc)
        
        echo ""
        echo "📊 [$current_test/$total_tests - ${progress}%] 评估模型: $model_dataset 在数据集: $test_dataset"
        echo "Evaluating on $test_dataset"
        
        # 运行WiSE-FT模型评估 - 使用与finetune相同的测试脚本，只使用第一个GPU避免并行问题
        FIRST_GPU=$(echo $GPU | cut -d',' -f1)
        result=$(CUDA_VISIBLE_DEVICES=$FIRST_GPU python test.py \
            --eval-datasets=$test_dataset \
            --load="$model_path" \
            --data-location=/data/liyuting/CL/mtil_datasets \
            2>&1 | grep "Top-1 accuracy:" | tail -1)
        
        if [ -z "$result" ]; then
            echo "⚠️ 评估失败 - 请检查模型和环境配置"
        else
            echo "$result"
        fi
        
        echo "✅ 完成评估: $model_dataset -> $test_dataset"
    done
    
done
