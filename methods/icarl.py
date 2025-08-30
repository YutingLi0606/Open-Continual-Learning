import os
import re
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import utils
import models.clip.clip as clip
from methods.zscl.helpers import batch
from training.finetune_zscl import finetune as pure_finetune
from evaluation.evaluate import evaluate, zeroshot_classifier
from training.finetune_zscl import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation


def iCaRL(args):
    assert args.dataset_order is not None, "order need to be provided"
    assert args.train_dataset is None
    assert args.load is None
    dataset_order = args.dataset_order
    ref_dataset = DynamicDataset(args) 
    args.train_dataset = dataset_order[0]
    print(args.train_dataset == "StanfordCars")
    if args.train_dataset in ["Aircraft", "MNIST"]:
        args.lr = 5e-5
    else:
        args.lr = 1e-5
    pure_finetune(args)
    ref_dataset.update(args, args.train_dataset)
    args.load = os.path.join(args.save, f"{dataset_order[0]}.pth")

    for i in np.arange(1, len(dataset_order)):
        train_dataset = dataset_order[i]
        args.train_dataset = train_dataset
        ref_images = ref_dataset.get()
        if args.train_dataset in ["Aircraft", "MNIST"]:
            args.lr = 5e-5
        else:
            args.lr = 1e-5
        finetune(args, ref_images)
        args.load = os.path.join(args.save, f"{dataset_order[i]}.pth")
        ref_dataset.update(args, args.train_dataset, args.load)
        
def finetune(args, ref_images):
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
    if args.load is not None:
        utils.torch_load(model, args.load)

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    template = dataset.template

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
        args.eval_datasets = [args.train_dataset]
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    assert args.train_mode == "whole"
    print("[Training mode] Both Encoders")
    exclude_params_name = ["logit_scale"]
    params = [
        v for k, v in model.named_parameters() if k not in exclude_params_name
    ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method

    if args.method == "icarl":
        # (Ref Model) get reference model
        print("[Method] iCaRL")
        if args.ref_model is None:
            print("[ref_model] Zero-shot")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        else:
            print(f"[ref_model] {args.ref_model}")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(
                ref_model, args.ref_model
            )
        ref_model = ref_model.cuda()
        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model.eval()

        # (Ref Text) get reference text and reference images
        ref_images_batch = batch(ref_images, args.batch_size)
        print(f"[Ref Sentences] {args.train_dataset}")
        ref_texts = texts
        ### use traing dataset template in iCarRL
            

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if eval_iterations is not None and iteration % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess)

        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        if args.method == "icarl":
            
            # -- get reference images --
            try:
                ref_img = next(ref_images_batch)
            except:
                ref_images_batch = batch(ref_images, args.batch_size)
                ref_img = next(ref_images_batch)
            ref_img = ref_img.cuda()


            with torch.no_grad():
                # -- get ref text embedding --
                ref_embeddings = ref_model(None, ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # -- get ref image embedding --
                ref_out = ref_model(ref_img, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            ref_out_current = model(ref_img, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            # -- loss --
            logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
            loss_lwa = distillation(logits_ref, logits_current, T=args.T)

            # feature-space mse
            if args.feature_mse:
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(ref_out, ref_out_current)

            # -- final loss --
            if args.image_loss:
                if args.weight_adjust:
                    loss = loss + 0.5 * loss_lwa 
                else:
                    loss = loss + 1.0 * loss_lwa 

            # transpose loss
            if args.text_loss:
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_lwa_2 = distillation(logits_ref_2, logits_current_2, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_lwa_2
                else:
                    loss += loss_lwa_2
            
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
            if args.method == "icarl":
                print("Loss iCaRL:", loss_lwa.item())

    # Saving model
    if args.save is not None:
        to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
    
class DynamicDataset():

    def __init__(self, args):
        self.ref_database = {}  # all data key = dataset_name; value is 4D tensor (num_image, 3, 224, 224)
        self.ref_names = [] # collect the name of the dataset 
        self.ref_model, _, self.test_preprocess = clip.load(args.model, jit=False)
        self.cur_dataset = None
        self.memory_size = args.memory_size

    def update(self, args, dataset, load=False):
        self.cur_dataset = dataset
        if not load: # first round
            new_dataset = self.getNewDataset(args)
            self.ref_database[self.cur_dataset] = new_dataset[:self.memory_size]
            self.ref_names.append(self.cur_dataset)
        else: # other rounds
            utils.torch_load(self.ref_model, load)
            self.reduceExampleSet()
            self.constructExampleSet(args)

    def reduceExampleSet(self):
        print("Reducing Example Set")
        K, t = self.memory_size, len(self.ref_names)+1
        m = K // t
        for dataset in self.ref_names:
            self.ref_database[dataset] = self.ref_database[dataset][:m]

    def constructExampleSet(self, args):
        # breakpoint()
        print("Constructing Example Set")
        self.ref_names.append(self.cur_dataset)
        new_dataset = torch.tensor(self.getNewDataset(args))
        image_feature = []
        num = new_dataset.shape[0]

        print("[Constructing] Calculating Distance")
        for ndx in tqdm(np.arange(num)):
            img = torch.unsqueeze(new_dataset[ndx], dim=0)
            img = img.cuda()
            img_feature = self.ref_model(img, None)
            image_feature.append(img_feature.cpu().detach().tolist())
        image_feature = torch.tensor(image_feature)
        image_feature = torch.squeeze(image_feature, dim=1)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        image_feature = np.array(image_feature.cpu().detach())
        image_feature_average = image_feature.mean(axis=0)

        K, t = self.memory_size, len(self.ref_names)
        m = K - K // t
        update_dataset = []
        cur_embedding_sum = None
        print("[Constructing] Collecting Examples")
        for k in tqdm(np.arange(min(m, len(image_feature)))):
            if not k:
                index = np.argmin(
                    np.sum((image_feature_average - image_feature)**2, axis=1)
                )
                cur_embedding_sum = image_feature[index]
                update_dataset.append((new_dataset.cpu())[index].tolist())
                image_feature = np.delete(image_feature, index, axis=0)
            else:
                index = np.argmin(
                    np.sum((
                        image_feature_average - (1/(k+1))*(image_feature + cur_embedding_sum)
                    )**2, axis=1)
                )
                cur_embedding_sum += image_feature[index]
                update_dataset.append((new_dataset.cpu())[index].tolist())
                image_feature = np.delete(image_feature, index, axis=0)
        
        self.ref_database[self.cur_dataset] = update_dataset

    def getNewDataset(self, args):
        dataset_class = getattr(datasets, self.cur_dataset)
        update_dataset = dataset_class(
            self.test_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        samples = [] 
        count = 0
        for sample in tqdm(update_dataset.train_dataset):
            if count == 30000:
                return samples
            count += 1
            samples.append(sample[0].tolist())
        return samples   

    def get(self):
        print("Getting Reference Images")
        value = list(self.ref_database.values())
        out = []
        for i in tqdm(value):
            out += i       
        return torch.tensor(out)
    