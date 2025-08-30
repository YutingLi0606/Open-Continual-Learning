import os
import clip
import json
import numpy as np
from tqdm import tqdm
import torch

import dataloader, utils
from dataloader.common import get_dataloader, maybe_dictionarize
from methods.zscl.modeling import create_zeroshot_classifier_head


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in classnames:
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)

    print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")


def evaluate(image_classifier, args, val_preprocess):
    if args.eval_datasets is None:
        return
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(dataloader, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset(image_classifier, dataset, args)




def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = "features"
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = "images"
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    batched_data = enumerate(dataloader)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0

        for i, data in tqdm(batched_data):
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            print(x.shape)
            y = data["labels"].to(device)
            logits, feature = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    metrics = {}
    metrics["top1"] = top1

    return metrics


def evaluate_fc(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    old_head = image_classifier.classification_head

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(dataloader, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        if args.dataset_shift:
            image_classifier.classification_head = create_zeroshot_classifier_head(
                args, dataset=dataset_name
            )

        results = eval_single_dataset(image_classifier, dataset, args)

        if "top1" in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if "worst" in key or "f1" in key.lower() or "pm0" in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ":" + key] = val

    image_classifier.classification_head = old_head

    # Save results
    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, "a+") as f:
            f.write(json.dumps(info) + "\n")
        print(f"Results saved to {args.results_db}.")
    else:
        print("Results not saved (to do so, use --results_db to specify a path).")

    return info

