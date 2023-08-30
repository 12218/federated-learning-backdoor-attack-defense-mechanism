import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import data, server
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cifar10_classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck"
    ]

    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('-c', '--conf', '--configuration', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = json.load(file)

    # load dataset
    _, test_loader = data.get_data(dir='./datasets/', conf=conf, return_type='dataloader')

    pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])

    # for imgs, labels in test_loader:

    #     poisoned_labels = labels.clone()
    #     for m in range(imgs.shape[0]):
    #         img = imgs[m].numpy()
    #         for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
    #             img[0][pos[i][0]][pos[i][1]] = 1.0
    #             img[1][pos[i][0]][pos[i][1]] = 0
    #             img[2][pos[i][0]][pos[i][1]] = 0
    #         poisoned_labels[m] = conf['malicious']['poison_label']

    #     for i in range(4):
    #         img_array = np.transpose(imgs[i], (1, 2, 0))

    #         plt.subplot(2, 2, i + 1)
    #         plt.imshow(img_array)
    #         plt.title('Real Label: {} - {}'.format(labels[i], cifar10_classes[labels[i]]))
    #     plt.tight_layout()
    #     plt.savefig('./images/poisoned_images.png')
    #     plt.show()

    #     break

    img_list = []
    for imgs, labels in test_loader:
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                img_list.append(imgs[i])

        if len(img_list) >= 4:
            break

    for i in range(4):
        img_array = np.transpose(img_list[i], (1, 2, 0))

        plt.subplot(2, 2, i + 1)
        plt.imshow(img_array)
        plt.title('Label: {} - {}'.format(2, 'bird'))
    plt.tight_layout()
    plt.savefig('./images/bird_images.png')
    plt.show()