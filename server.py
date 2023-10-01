import model
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

class Server():
    def __init__(self, conf, test_dataset) -> None:
        self.conf = conf
        self.global_model = model.get_model(model_type=self.conf['server']['model_type'])
        self.test_dataset = test_dataset

        self.test_loader = self.dataloader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dataloader(self):
        # 获取数据集的总长度
        total_samples = len(self.test_dataset)

        # 计算一半的数据数量
        half_samples = total_samples // 8

        # 使用切片获取一半的数据
        half_dataset = torch.utils.data.Subset(self.test_dataset, indices=range(half_samples))
        return DataLoader(half_dataset,
                    batch_size=self.conf['server']['batch_size'],
                    shuffle=True)
    
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
    
    def model_evaluation(self):
        self.global_model.eval()

        loss_function = nn.CrossEntropyLoss()

        loss_sum = 0
        correct_num = 0
        sample_num = 0

        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            output = self.global_model(imgs)

            loss = loss_function(output, labels)
            loss_sum += loss

            prediction = torch.max(output, 1)
            correct_num += (labels == prediction[1]).sum()

            sample_num += labels.shape[0]

        accuracy = correct_num / sample_num

        return accuracy, loss_sum

    def model_poisoned_dataset_evaluation(self):
        self.global_model.eval()

        loss_function = nn.CrossEntropyLoss()

        loss_sum = 0
        correct_num = 0
        sample_num = 0

        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])

        for imgs, labels in self.test_loader:
            poisoned_labels = labels.clone()
            for m in range(imgs.shape[0]):
                img = imgs[m].numpy()
                for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                    img[0][pos[i][0]][pos[i][1]] = 1.0
                    img[1][pos[i][0]][pos[i][1]] = 0
                    img[2][pos[i][0]][pos[i][1]] = 0
                poisoned_labels[m] = self.conf['malicious']['poison_label']

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            output = self.global_model(imgs)

            loss = loss_function(output, labels)
            loss_sum += loss

            prediction = torch.max(output, 1)
            poisoned_labels = poisoned_labels.to(self.device)
            correct_num += (poisoned_labels == prediction[1]).sum()

            sample_num += labels.shape[0]

        accuracy = correct_num / sample_num

        return accuracy, loss_sum