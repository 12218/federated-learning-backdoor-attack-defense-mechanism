from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse, json

def get_data(dir, conf, return_type):
    """
    Return Cifar-10 datasets or dataloaders

    dir: the path to store the cifar dataset
    conf: configuration dictionary
    return_type: return dataset or dataloader
    """

    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))
    ])

    train_dataset = CIFAR10(dir, train=True, transform=train_trans, download=True)
    test_dataset = CIFAR10(dir, train=False, transform=test_trans, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=conf['client']['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=conf['client']['batch_size'], shuffle=True)

    if return_type == 'dataset':
        return train_dataset, test_dataset
    elif return_type == 'dataloader':
        return train_loader, test_loader
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('-c', '--conf', '--configuration', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = json.load(file)

    train_set, test_set = get_data(dir='./datasets/', conf=conf, return_type='dataset')
    train_loader, test_loader = get_data(dir='./datasets/', conf=conf, return_type='dataloader')

    print(type(train_set), type(test_set))
    print(type(train_loader), type(test_loader))