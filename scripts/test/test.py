#!/usr/bin/env python3
"""
专门用于测试finetune模型的脚本
使用正确的zeroshot评估方法
"""

import sys
import os
import argparse
import clip
import torch
from src import utils
from evaluation.evaluate import eval_single_dataset, zeroshot_classifier, zeroshot_eval
from dataloader.common import get_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ViT-B/16')
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--load', required=True)
    parser.add_argument('--data-location', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-size-eval', type=int, default=128)
    
    args = parser.parse_args()
    
    # 加载模型
    model, _, val_preprocess = clip.load(args.model, jit=False)
    if args.load:
        utils.torch_load(model, args.load)
    
    model.eval()
    
    # 加载数据集
    dataset_class = getattr(datasets, args.eval_datasets)
    dataset = dataset_class(
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
    
    # 使用zeroshot评估方法
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )
    
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None
    )
    print(111)
    
    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)
    
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

if __name__ == "__main__":
    main()