import os
import copy
import numpy as np
from random import random

import torch
import torch.nn as nn
import torchvision.models as models

import utils
import evaluation
import models.clip.clip as clip
from args import parse_arguments
from methods.zscl.modeling import create_image_classifier
from methods.moe import AutoEncoder, few_shot_AutoEncoder
from evaluation.evaluate_moe import evaluate as evaluate_moe
from training.finetune_moe import finetune as finetune_moe
from training.finetune_icarl import iCaRL as finetune_icarl
from training.finetune_zscl import finetune as finetune_zscl
from training.finetune_dual_rail import finetune as finetune_dual_rail
from training.finetune_primal_rail import finetune as finetune_primal_rail



def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1


def main(args):
    print(args)
    utils.seed_all(args.seed)

    # Evaluation only mode
    if args.eval_only:
        model, _, val_preprocess = clip.load(args.model, jit=False, args=args if hasattr(args, 'autorouter') else None)
        
        if args.load:
            if hasattr(args, 'wise_ft') and args.wise_ft:
                print("Use wise-ft.")
                model_0 = copy.deepcopy(model)
            utils.torch_load(model, args.load)
            if hasattr(args, 'wise_ft') and args.wise_ft:
                model = merge(model_0, model, alpha=args.alpha)
        
        elif args.save:
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(checkpoint_pth, model)
        
        # MoE evaluation
        if hasattr(args, 'method') and args.method == 'moe':
            if hasattr(args, 'load_autochooser') and args.load_autochooser and hasattr(args, 'autorouter') and args.autorouter:
                pretrained_alexnet = models.alexnet(pretrained=True)
                feature_extractor = AutoEncoder.Alexnet_FE(pretrained_alexnet).cuda()
                Autoencoder_list = nn.ModuleList()
                for i in range(args.task_num + 1):  # more for zero-shot chosen
                    model_autoencoder = AutoEncoder.Autoencoder(256 * 13 * 13)
                    Autoencoder_list.append(model_autoencoder)
                utils.torch_load(Autoencoder_list, args.load_autochooser)
                Autoencoder_list = Autoencoder_list.cuda()
            else:
                feature_extractor = None
                Autoencoder_list = None
            evaluate_moe(model, feature_extractor, Autoencoder_list, args, val_preprocess)
        else:
            # Standard evaluation
            evaluation.evaluate(model, args, val_preprocess)
    
    # Training mode
    else:
        # MoE training
        if hasattr(args, 'method') and args.method == 'moe':
            if hasattr(args, 'train_chooser'):
                if args.train_chooser:
                    if hasattr(args, 'few_shot') and args.few_shot > 0:
                        print('----------------------train few-shot chooser----------------------')
                        chooser_of_few_shot = few_shot_AutoEncoder.few_shot_AutoEncoder(args)
                    else:
                        print('----------------------train full-shot chooser----------------------')
                        chooser = AutoEncoder.AutoEncoder(args)
                else:
                    print('----------------------finetune model----------------------')
                    model = finetune_moe(args)
            else:
                print('----------------------finetune model----------------------')
                model = finetune_moe(args)
        
        # iCaRL method
        elif hasattr(args, 'method') and args.method == "icarl":
            model = finetune_icarl(args)
        
        # RAIL methods
        elif hasattr(args, 'method') and args.method == "rail":
            if hasattr(args, 'rail_version') and args.rail_version == 'primal':
                print("Using Primal RAIL method")
                model = finetune_primal_rail(args)
            else:
                print("Using Dual RAIL method")
                model = finetune_dual_rail(args)
        
        # ZSCL and other methods
        else:
            model = finetune_zscl(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)