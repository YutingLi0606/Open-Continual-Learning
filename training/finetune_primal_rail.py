"""
Multi-domain transfer CLIP under Primal RAIL version
"""

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils, dataloader
import models.clip.clip as clip
from evaluation.evaluate import evaluate
from methods.rail import clip_classifier, cls_acc, ContinualCLIPAdaptor



def finetune(args):
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    
    # Set random seed
    if hasattr(args, 'seed'):
        utils.seed_all(args.seed)
        
    # Set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get dataset sequence
    dataset_sequence = args.dataset_sequence
    print("Multi-task dataset sequence: ", dataset_sequence)

    """
    Loading model
    """
    print('Loading pretrained CLIP model...')
    # Create a temporary model to get transforms
    temp_model, train_transform, val_preprocess = clip.load(args.model, device=args.device, jit=False)
    
    # Build all datasets with proper transforms
    merged_classnames = []
    dataset_objects = []
    
    for dataset_name in dataset_sequence:
        dataset_class = getattr(dataloader, dataset_name)
        dataset = dataset_class(
            train_transform,  # Use train transform
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        dataset_objects.append(dataset)
        merged_classnames += dataset.classnames

    # Results
    fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
    adapter_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
    
    args.previous_class_num = 0
    current_class_names = []
    R = None

    # Create the actual continual clip adaptor
    continual_clip_adaptor = ContinualCLIPAdaptor(args)
    continual_clip_adaptor.clip_model.eval()

    """
    Training on dataset sequence
    """
    for task_id, (train_dataset_name, dataset) in enumerate(zip(dataset_sequence, dataset_objects)):
        print(f"------------------ Start training on task-{task_id + 1}: {train_dataset_name}. ---------------------")

        current_class_names += dataset.classnames
        args.increment = len(dataset.classnames)
        args.current_class_num = len(current_class_names)

        # Get train loader
        train_loader = dataset.train_loader
        R = continual_clip_adaptor.analytic_adaption(task_id, args, train_loader, R)

        args.trained_class_num = args.current_class_num

        """
        Testing stage: test on every dataset (both trained & untrained) after training on each dataset
        """
        if hasattr(args, 'eval_last') and args.eval_last and task_id < len(dataset_sequence) - 1:
            continue

        tested_cls_num = 0
        for test_id, (test_dataset_name, test_dataset) in enumerate(zip(dataset_sequence, dataset_objects)):
            if hasattr(args, 'eval_adapter') and args.eval_adapter and test_id > task_id:
                continue

            print(f"Evaluating on dataset-{test_id + 1}: {test_dataset_name}")
            
            test_loader = test_dataset.test_loader

            template = ['a photo of a {}.']
            clip_weights = clip_classifier(merged_classnames, template, continual_clip_adaptor.clip_model, device=args.device)

            top1, top5, test_num = 0.0, 0.0, 0.0
            fusion_top1, fusion_top5 = 0.0, 0.0

            for inputs, targets in tqdm(test_loader, 
                                      desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset_name}',
                                      total=len(test_loader), unit='batch'):
                test_num += inputs.size(0)

                inputs, targets = inputs.to(args.device), targets.to(args.device)
                targets = targets + tested_cls_num

                with torch.no_grad():
                    outputs = continual_clip_adaptor.zero_shot(inputs, clip_weights)  # (B, C_all)
                    outputs = F.softmax(outputs, dim=-1)

                predict_cls = torch.argmax(outputs, dim=-1)

                # Zero-shot acc
                acc1, acc5 = cls_acc(outputs, targets, topk=(1, 5))
                top1 += acc1
                top5 += acc5

                # Select ID samples belonging to learned domains by zero-shot
                mask = predict_cls < args.current_class_num
                if torch.sum(mask) > 0:
                    samples_to_adapt = inputs[mask]
                    with torch.no_grad():
                        outputs_adapted = continual_clip_adaptor(samples_to_adapt)

                    padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                    outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)

                    outputs[mask] = (1 - args.fusion_weight) * outputs[mask] + args.fusion_weight * outputs_adapted

                # Fusion acc
                fusion_acc1, fusion_acc5 = cls_acc(outputs, targets, topk=(1, 5))
                fusion_top1 += fusion_acc1
                fusion_top5 += fusion_acc5

            top1, top5 = (top1 / test_num) * 100, (top5 / test_num) * 100
            print(f"Zero-shot top-1 acc for dataset-{test_id + 1}: {test_dataset_name}: {top1}")

            fusion_acc = (fusion_top1 / test_num) * 100
            print(f"***** Fusion top-1 acc for dataset-{test_id + 1}: {test_dataset_name}: {fusion_acc} *****")
            fusion_acc_table[task_id, test_id] = fusion_acc

            tested_cls_num += len(test_dataset.classnames)

    # Calculate final metrics
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    
    upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
    masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
    transfer_acc = np.mean(masked_matrix, axis=0)
    transfer_avg_acc = np.mean(transfer_acc)
    avg_acc = np.mean(fusion_acc_table, axis=0)
    avg_avg_acc = np.mean(avg_acc)
    
    print('Average transfer acc: ', transfer_avg_acc)
    print('Average average acc: ', avg_avg_acc)
    print('Average last acc: ', np.mean(fusion_acc_table[-1, :]))
    
    # Print detailed table
    print("\nDetailed Accuracy Table:")
    print("Task ->", end="")
    for i, name in enumerate(dataset_sequence):
        print(f"{name:>10}", end="")
    print()
    
    for i in range(len(dataset_sequence)):
        print(f"Task-{i+1:>2}:", end="")
        for j in range(len(dataset_sequence)):
            print(f"{fusion_acc_table[i,j]:>10.2f}", end="")
        print()
    
    # Save results if specified
    if hasattr(args, 'save') and args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        np.save(os.path.join(args.save, 'fusion_acc_table.npy'), fusion_acc_table)
        with open(os.path.join(args.save, 'results.txt'), 'w') as f:
            f.write(f'Average transfer acc: {transfer_avg_acc:.2f}\n')
            f.write(f'Average average acc: {avg_avg_acc:.2f}\n')
            f.write(f'Average last acc: {np.mean(fusion_acc_table[-1, :]):.2f}\n')
    
    return continual_clip_adaptor

if __name__ == "__main__":
    pass