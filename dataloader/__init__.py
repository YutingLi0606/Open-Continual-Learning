from .imagenet import ImageNet
from .cc import conceptual_captions
from .cifar import CIFAR101, CIFAR102

# Small
from .collections import (
    TinyImagenet,
    CIFAR10,
    CIFAR100,
    DTD,
    MNIST,
    SUN397,
    Aircraft,
    Caltech101,
    EuroSAT,
    Flowers,
    Food,
    OxfordPet,
    StanfordCars,
)

dataset_list = [
    TinyImagenet,
    Aircraft,
    Caltech101,
    CIFAR10,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
]

def show_datasets():
    print("Total: ", len(dataset_list))
    print("Dataset: (train_len, test_len, num_classes)")
    for dataset in dataset_list:
        d = dataset(None)
        print(f"{d.name}: ", d.stats())
        for i in range(3):
            print(f"T[{i}]: ", d.template(d.classnames[i]))
        


