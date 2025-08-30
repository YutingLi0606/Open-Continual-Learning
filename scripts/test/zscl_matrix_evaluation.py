import os
import re
import sys
import csv
import json
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional



class ZSCLMatrixEvaluator:
    
    def __init__(self):
        self.datasets = [
            "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", 
            "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"
        ]
        self.accuracy_matrix = np.zeros((11, 11))
        
    def cal_ZSCL_metrics(self, acc_matrix):
        """
        计算ZSCL指标，包括Transfer、Average和Last性能
        """
        acc_matrix = np.array(acc_matrix)
        acc_matrix *= 100  # 转换为百分比
        
        avg = acc_matrix.mean(axis=0)  # 每列的平均值
        last = np.array(acc_matrix[-1, :])  # 最后一行
        
        # Transfer: 对每个数据集，计算之前训练的所有模型在该数据集上的平均性能
        transfer = np.array([
            np.mean([acc_matrix[j, i] for j in range(i)]) if i > 0 else 0
            for i in range(acc_matrix.shape[1])
        ])
        
        g = lambda x: np.around(x.mean(), decimals=1) if len(x) > 0 else -1
        f = lambda x: [np.around(i, decimals=1) for i in x]
        s = lambda x: np.around(x.sum(), decimals=1) if len(x) > 0 else -1
        return {
            "transfer": {"transfer": f(transfer)}, 
            "avg": {"avg": f(avg)}, 
            "last": {"last": f(last)}, 
            "results_mean": {
                "transfer": s(transfer)/10, # Transfer从第2个任务开始计算
                "avg": g(avg), 
                "last": g(last)
            }
        }
    
    def run_test_script_and_capture_matrix(self, test_script_path: str = "scripts/test_zscl.sh") -> np.ndarray:
        """
        运行ZSCL测试脚本并捕获11x11矩阵结果
        """
        if not os.path.exists(test_script_path):
            raise FileNotFoundError(f"测试脚本不存在: {test_script_path}")
        
        log_file = f"zscl_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ['bash', test_script_path], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    print(line.strip())  # 实时显示输出
                    f.write(line)  # 同时写入日志文件
                
                process.wait()
            
            # 解析日志文件生成矩阵
            matrix = self.parse_log_to_matrix(log_file)
            
            return matrix
            
        except Exception as e:
            print(f"运行测试脚本时出错: {e}")
            return np.zeros((11, 11))
    
    def parse_log_to_matrix(self, log_file: str) -> np.ndarray:
        """
        解析ZSCL测试日志文件，提取11x11准确率矩阵
        """
        matrix = np.zeros((11, 11))
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 解析逻辑：查找加载checkpoint和评估结果的模式
        lines = content.split('\n')
        
        current_model_idx = -1
        current_dataset_idx = -1
        
        print("🔍 解析日志文件...")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测加载的模型checkpoint
            if "Checkpoint loaded from" in line and ".pth" in line:
                # 提取模型名称 - ZSCL模型按数据集名称保存
                model_match = re.search(r'/([^/]+)\.pth', line)
                if model_match:
                    model_name = model_match.group(1)
                    if model_name in self.datasets:
                        current_model_idx = self.datasets.index(model_name)
                        print(f"   检测到模型: {model_name} (索引: {current_model_idx})")
            
            # 检测正在评估的数据集
            elif line.startswith("Evaluating on "):
                dataset_match = re.search(r"Evaluating on (\w+)", line)
                if dataset_match:
                    dataset_name = dataset_match.group(1)
                    if dataset_name in self.datasets:
                        current_dataset_idx = self.datasets.index(dataset_name)
                        print(f"   检测到数据集: {dataset_name} (索引: {current_dataset_idx})")
            
            # 检测准确率结果
            elif line.startswith("Top-1 accuracy:") and current_model_idx >= 0 and current_dataset_idx >= 0:
                acc_match = re.search(r"Top-1 accuracy: ([\d.]+)", line)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    # 如果准确率已经是百分比格式，就除以100；如果是小数格式，保持不变
                    if accuracy > 1.0:
                        accuracy = accuracy / 100.0
                    matrix[current_model_idx, current_dataset_idx] = accuracy
                    print(f"   记录准确率: 模型{current_model_idx}->数据集{current_dataset_idx} = {accuracy:.4f}")
            
            i += 1
        
        print(f"✅ 解析完成，非零元素数量: {np.count_nonzero(matrix)}")
        return matrix
    
    def save_accuracy_matrix_csv(self, matrix: np.ndarray, output_path: str):
        """
        将准确率矩阵保存为CSV文件
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            header = ["Model\\Dataset"] + self.datasets
            writer.writerow(header)
            
            # 写入每一行数据
            for i, model_name in enumerate(self.datasets):
                row = [model_name] + [f"{matrix[i, j]*100:.2f}" for j in range(11)]
                writer.writerow(row)

    
    def save_zscl_metrics_csv(self, metrics: Dict, experiment_name: str, output_path: str):
        """
        将ZSCL指标保存为CSV文件
        """
        # CSV表头：第一列为指标类型，第二列为方法名，后面是各数据集结果，最后一列是平均值
        csv_headers = ["", "Method"] + self.datasets + ["Average"]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow(csv_headers)
            
            # 提取数据
            transfer_data = metrics["transfer"]["transfer"]
            avg_data = metrics["avg"]["avg"] 
            last_data = metrics["last"]["last"]

            # 计算每行的平均值
            transfer_avg = np.mean(transfer_data)
            avg_avg = np.mean(avg_data)
            last_avg = np.mean(last_data)
            
            # 格式化函数
            format_acc = lambda x: f"{x:.2f}"

            # 写入Transfer行
            transfer_row = ["Transfer", experiment_name] + [format_acc(x) for x in transfer_data] + [format_acc(transfer_avg)]
            writer.writerow(transfer_row)
            
            # 写入Average行
            avg_row = ["Average", experiment_name] + [format_acc(x) for x in avg_data] + [format_acc(avg_avg)]
            writer.writerow(avg_row)
            
            # 写入Last行
            last_row = ["Last", experiment_name] + [format_acc(x) for x in last_data] + [format_acc(last_avg)]
            writer.writerow(last_row)
    
    def run_full_evaluation(self, test_script_path: str = "scripts/test_zscl.sh", 
                           experiment_name: str = "ZSCL_Matrix_Evaluation",
                           matrix_csv: str = "accuracy_matrix.csv",
                           metrics_csv: str = "zscl_metrics.csv",
                           checkpoint_dir: str = None):
        """
        运行完整的ZSCL矩阵评估
        """
        print(f"🚀 开始ZSCL矩阵评估...")
        print(f"   测试脚本: {test_script_path}")
        print(f"   实验名称: {experiment_name}")
        
        # 运行测试脚本并获取矩阵
        matrix = self.run_test_script_and_capture_matrix(test_script_path)
        
        if matrix.sum() == 0:
            print("警告: 未收集到有效的准确率数据")
            return

        # 如果指定了checkpoint目录，则将文件保存到该目录
        if checkpoint_dir:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            matrix_csv = os.path.join(checkpoint_dir, os.path.basename(matrix_csv))
            metrics_csv = os.path.join(checkpoint_dir, os.path.basename(metrics_csv))

        # 保存准确率矩阵
        self.save_accuracy_matrix_csv(matrix, matrix_csv)
        
        # 计算并保存ZSCL指标
        metrics = self.cal_ZSCL_metrics(matrix)
        self.save_zscl_metrics_csv(metrics, experiment_name, metrics_csv)
        
        print(f"\n✅ 评估结果已保存:")
        print(f"   矩阵文件: {matrix_csv}")
        print(f"   指标文件: {metrics_csv}")
        
        # 输出关键指标摘要
        print(f"\n📊 关键指标摘要:")
        print(f"   🔄 Transfer Average: {metrics['results_mean']['transfer']:.2f}")
        print(f"   📊 Average Performance: {metrics['results_mean']['avg']:.2f}")
        print(f"   🎯 Last Performance: {metrics['results_mean']['last']:.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ZSCL矩阵评估")
    parser.add_argument("--test-script", default="scripts/test_zscl.sh", help="测试脚本路径")
    parser.add_argument("--experiment-name", default="ZSCL_Matrix_Evaluation", help="实验名称")
    parser.add_argument("--matrix-csv", default="accuracy_matrix.csv", help="准确率矩阵输出文件")
    parser.add_argument("--metrics-csv", default="zscl_metrics.csv", help="ZSCL指标输出文件")
    parser.add_argument("--checkpoint-dir", default=None, help="checkpoint目录，结果文件将保存到此目录")
    parser.add_argument("--model-ckpt-path", default="ckpt/exp_zscl", help="测试脚本中使用的模型checkpoint路径")
    
    args = parser.parse_args()
    
    evaluator = ZSCLMatrixEvaluator()
    evaluator.run_full_evaluation(
        args.test_script, 
        args.experiment_name, 
        args.matrix_csv, 
        args.metrics_csv,
        args.checkpoint_dir
    )


if __name__ == "__main__":
    main()