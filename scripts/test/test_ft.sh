#!/bin/bash

# ZSCL测试脚本 - 11x11矩阵评估
# 测试每个训练好的模型在所有数据集上的性能

set -e

# 激活moe环境
source ~/.bashrc
conda activate moe

# 参数设置
MODEL_CKPT_PATH="${1:-ckpt/exp_finetune}"
GPU="${2:-0,1,2,3}"
TEST_MODE="${3:-whole}"

# 数据集列表
datasets=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)

echo "🧪 ZSCL 11x11矩阵评估测试"
echo "=========================="
echo "模型路径: $MODEL_CKPT_PATH"
echo "GPU: $GPU"
echo "测试模式: $TEST_MODE"
echo "数据集数量: ${#datasets[@]}"
echo ""

# 检查模型checkpoint目录是否存在
if [ ! -d "$MODEL_CKPT_PATH" ]; then
    echo "❌ 错误: 模型checkpoint目录不存在: $MODEL_CKPT_PATH"
    exit 1
fi

# 切换到mtil目录（如果存在）
if [ -d "mtil" ]; then
    cd mtil
    echo "📂 切换到mtil目录"
fi

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
        
        # 运行Finetune模型评估 - 使用专门的测试脚本，只使用第一个GPU避免并行问题
        FIRST_GPU=$(echo $GPU | cut -d',' -f1)
        result=$(CUDA_VISIBLE_DEVICES=$FIRST_GPU python test_finetune_model.py \
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
    
    echo ""
    echo "✅ 模型 $model_dataset 评估完成"
    echo "----------------------------------------"
done

echo ""
echo "🎉 ZSCL 11x11矩阵评估完成!"
echo "所有模型在所有数据集上的评估已完成"
echo ""