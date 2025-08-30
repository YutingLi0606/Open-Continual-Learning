

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils, dataloader
import models.clip.clip as clip
from evaluation.evaluate import evaluate

from methods.rail import kernel_ridge_regression, clip_classifier, encode_images, cls_acc

def finetune(args):
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    
    # Set random seed
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    # Load pretrained CLIP model
    print('Loading pretrained CLIP model...')
    model, train_preprocess, val_preprocess = clip.load(args.model, args.device, jit=False)
    
    if args.load is not None:
        utils.torch_load(model, args.load)
    
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.eval()

    # Get dataset sequence
    dataset_sequence = args.dataset_sequence
    print("Multi-task dataset sequence: ", dataset_sequence)

    # Build all datasets and collect classnames
    merged_classnames = []
    dataset_objects = []
    
    for dataset_name in dataset_sequence:
        dataset_class = getattr(dataloader, dataset_name)
        dataset = dataset_class(
            train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        dataset_objects.append(dataset)
        merged_classnames += dataset.classnames

    # Initialize metrics
    fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
    
    # Initialize KRR
    krr = kernel_ridge_regression(lamda=args.krr_lambda, gamma=args.krr_gamma)
    trained_class_num = 0
    current_class_names = []
    feature_memory = None
    y_memory = None
    
    # Default template for zero-shot
    template = ['a photo of a {}.']

    """
    Training on dataset sequence
    """
    for task_id, (dataset_name, dataset) in enumerate(zip(dataset_sequence, dataset_objects)):
        print(f"------------------ Start training on task-{task_id + 1}: {dataset_name}. ---------------------")

        current_class_names += dataset.classnames
        increment = len(dataset.classnames)
        current_class_num = len(current_class_names)

        # Extract training features
        current_train_features = []
        current_train_one_hot_labels = []
        
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(dataset.train_loader, 
                                                    desc=f'Extracting training features for {dataset_name}',
                                                    total=len(dataset.train_loader))):
                target = target + trained_class_num  # Adjust label offset
                images, target = images.cuda(), target.cuda()
                # Use model.module to access the original CLIP model methods
                img_embeddings = encode_images(model.module, images)
                
                # Create one-hot labels for current number of classes
                train_labels_one_hot = F.one_hot(target, current_class_num).float()
                
                current_train_features.append(img_embeddings)
                current_train_one_hot_labels.append(train_labels_one_hot)

        current_train_features = torch.cat(current_train_features, dim=0)
        current_train_one_hot_labels = torch.cat(current_train_one_hot_labels, dim=0).cpu().numpy()

        # Update feature memory and label memory
        if task_id == 0:
            feature_memory = current_train_features
            y_memory = current_train_one_hot_labels
        else:
            # Concatenate features
            feature_memory = torch.cat([feature_memory, current_train_features], dim=0)
            
            # Extend previous labels with zeros for new classes
            y_extended = np.concatenate([y_memory, np.zeros((y_memory.shape[0], increment))], axis=1)
            
            # Concatenate with new labels
            y_memory = np.concatenate([y_extended, current_train_one_hot_labels], axis=0)

        # Train KRR
        alpha = krr.train(feature_memory, y_memory)
        trained_class_num = current_class_num
        
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

            # Build CLIP zero-shot classifier
            clip_weights = clip_classifier(merged_classnames, template, model.module, device=args.device)
            
            top1, fusion_top1, adapt_top1 = 0.0, 0.0, 0.0
            test_num = 0.0

            with torch.no_grad():
                for inputs, targets in tqdm(test_dataset.test_loader, 
                                          desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset_name}',
                                          total=len(test_dataset.test_loader)):
                    test_num += inputs.size(0)
                    inputs, targets = inputs.cuda(), targets.cuda()
                    targets = targets + tested_cls_num  # Adjust target labels
                    
                    # Use model.module to access the original CLIP model methods
                    test_features = encode_images(model.module, inputs)
                    
                    # Zero-shot prediction
                    outputs = 100. * test_features @ clip_weights
                    outputs = F.softmax(outputs, dim=-1)
                    
                    # Fusion prediction (KRR + CLIP)
                    if hasattr(args, 'fusion_weight') and args.fusion_weight > 0:
                        # Select samples that are predicted to belong to learned classes
                        predict_cls = torch.argmax(outputs, dim=-1)
                        mask = predict_cls < current_class_num
                        
                        if torch.sum(mask) > 0:
                            samples_to_adapt = test_features[mask]
                            outputs_adapted = krr.predict(samples_to_adapt, feature_memory)
                            outputs_adapted = torch.tensor(outputs_adapted, device=args.device, dtype=torch.float)
                            
                            # Zero-padding to match full dimension
                            padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                            outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)
                            
                            # Fusion
                            outputs[mask] = (1 - args.fusion_weight) * outputs[mask] + args.fusion_weight * outputs_adapted
                    
                    fusion_acc1 = cls_acc(outputs, targets)
                    fusion_top1 += fusion_acc1[0]
                    
                    # Pure KRR prediction (only for trained tasks)
                    if test_id <= task_id:
                        outputs_krr = krr.predict(test_features, feature_memory)
                        outputs_krr = torch.tensor(outputs_krr, device=args.device, dtype=torch.float)
                        adapt_acc1 = cls_acc(outputs_krr, targets)
                        adapt_top1 += adapt_acc1[0]
                    
                    # Zero-shot accuracy
                    zs_acc1 = cls_acc(outputs, targets)
                    top1 += zs_acc1[0]

            # Calculate final accuracies
            top1 = (top1 / test_num) * 100
            fusion_acc = (fusion_top1 / test_num) * 100
            adapt_acc = (adapt_top1 / test_num) * 100 if test_num > 0 else 0
            
            print(f"Zero-shot top-1 acc for dataset-{test_id + 1} ({test_dataset_name}): {top1:.2f}")
            print(f"Fusion top-1 acc for dataset-{test_id + 1} ({test_dataset_name}): {fusion_acc:.2f}")
            print(f"KRR top-1 acc for dataset-{test_id + 1} ({test_dataset_name}): {adapt_acc:.2f}")
            
            fusion_acc_table[task_id, test_id] = fusion_acc
            tested_cls_num += len(test_dataset.classnames)

    # Calculate final metrics
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    
    # Transfer accuracy (upper triangle excluding diagonal)
    upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
    masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
    transfer_acc = np.mean(masked_matrix, axis=0)
    transfer_avg_acc = np.mean(transfer_acc)
    
    # Average accuracy
    avg_acc = np.mean(fusion_acc_table, axis=0)
    avg_avg_acc = np.mean(avg_acc)
    
    # Last task accuracy
    last_acc = np.mean(fusion_acc_table[-1, :])
    
    print(f'Average transfer acc: {transfer_avg_acc:.2f}')
    print(f'Average average acc: {avg_avg_acc:.2f}')
    print(f'Average last acc: {last_acc:.2f}')
    
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
            f.write(f'Average last acc: {last_acc:.2f}\n')
    
    return model  # Return the trained model

if __name__ == "__main__":
    pass