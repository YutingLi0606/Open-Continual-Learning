#!/bin/bash

# ZSCL矩阵评估脚本
# 运行11x11完整评估并生成矩阵CSV和指标CSV

# 参数设置
EXPERIMENT_NAME="${1:-wise_ft_evaluation}"
SAVE_PATH="${2:-ckpt/exp_wiseft}"
MATRIX_CSV="${3:-accuracy_matrix.csv}"
METRICS_CSV="${4:-wiseft_metrics.csv}"
TEST_SH="${5:-scripts/test/test_wiseft.sh}"

start_time=$(date +%s)

# 运行矩阵评估
python scripts/test/zscl_matrix_evaluation.py \
    --test-script "$TEST_SH" \
    --experiment-name "$EXPERIMENT_NAME" \
    --matrix-csv "$MATRIX_CSV" \
    --metrics-csv "$METRICS_CSV" \
    --checkpoint-dir "$SAVE_PATH"


MATRIX_FILE="$SAVE_PATH/$MATRIX_CSV"
if [[ -f "$MATRIX_FILE" ]]; then
    echo "✅ 准确率矩阵: $MATRIX_FILE"
    echo ""
    echo "📋 矩阵预览 (前5行5列):"
    echo "------------------------"
    head -6 "$MATRIX_FILE" | cut -d',' -f1-6
    echo "------------------------"
else
    echo "❌ 准确率矩阵文件未生成"
fi

echo ""

METRICS_FILE="$SAVE_PATH/$METRICS_CSV"
if [[ -f "$METRICS_FILE" ]]; then
    echo "✅ ZSCL指标: $METRICS_FILE"
    echo ""
    echo "📈 ZSCL指标结果:"
    echo "=================="
    cat "$METRICS_FILE"
    echo ""
    
    # 提取关键数值
    transfer_avg=$(awk -F',' 'NR==2 {print $NF}' "$METRICS_FILE")
    average_avg=$(awk -F',' 'NR==3 {print $NF}' "$METRICS_FILE")
    last_avg=$(awk -F',' 'NR==4 {print $NF}' "$METRICS_FILE")
    
    echo "📊 关键指标摘要:"
    echo "  🔄 Transfer Average: $transfer_avg"
    echo "  📊 Average Performance: $average_avg" 
    echo "  🎯 Last Performance: $last_avg"
else
    echo "❌ ZSCL指标文件未生成"
fi

# 清理日志文件
echo ""
echo "🧹 清理临时文件..."
rm -f zscl_test_results_*.log

echo ""
echo "🎉 ZSCL矩阵评估完成!"
echo ""
echo "💡 使用说明:"
echo "   $0 \"实验名\" \"保存路径\" \"矩阵.csv\" \"指标.csv\""
echo ""
echo "📁 输出文件说明:"
echo "   • $MATRIX_CSV: 11x11准确率矩阵 (行=模型, 列=数据集)"
echo "   • $METRICS_CSV: Transfer/Average/Last指标"
echo "   • 文件保存到: $SAVE_PATH"
echo ""