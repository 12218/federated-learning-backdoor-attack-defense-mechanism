from datetime import datetime
from matplotlib import pyplot as plt
import model
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import numpy as np

class Client():
    def __init__(self, conf, train_dataset, id) -> None:
        self.conf = conf
        self.local_model = model.get_model(model_type=conf['client']['model_type'])
        self.train_dataset = train_dataset
        self.id = id

        self.train_loader = self.dataloader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dataloader(self):
        index_list = list(range(len(self.train_dataset))) # get the indices of the whole dataset
        local_data_len = int(len(self.train_dataset) / self.conf['num_workers']) # get the length of dataset on every client
        local_data_index_list = index_list[self.id * local_data_len: (self.id + 1) * local_data_len] # get a indices list of dataset for each client

        return DataLoader(self.train_dataset,
                          batch_size=self.conf['client']['batch_size'],
                          sampler=SubsetRandomSampler(local_data_index_list)) # select subset for each client from training dataset
    
    def local_train(self, global_model, global_epoch):
        self.local_model.train()
        # copy the parameters from global model to local model
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        learning_rate = self.conf['client']['lr']
        for i in range(len(self.conf['client']['lr_decrease_epochs'])):
            if global_epoch > self.conf['client']['lr_decrease_epochs'][i]:
                learning_rate *= 0.5
            else:
                continue

        # optimizer = torch.optim.SGD(self.local_model.parameters(),
        #                             lr=self.conf['client']['lr'],
        #                             momentum=self.conf['client']['momentum'])
        optimizer = torch.optim.Adam(self.local_model.parameters(),
                                    lr=learning_rate)
        
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(self.conf['client']['epoch']):
            loss_sum = 0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.local_model(imgs)

                loss = loss_function(output, labels)
                loss.backward()
                loss_sum += loss

                optimizer.step()

            print('Client {}: Epoch: {} - Loss: {}'.format(self.id, epoch + 1, loss_sum))

        print('Learning Rate: {}'.format(learning_rate))
        difference = {} # the difference between local model and global model

        for name, param in self.local_model.state_dict().items():
            difference[name] = param - global_model.state_dict()[name]

        return difference
    
    def local_train_malicious(self, global_model, global_epoch):
        # copy the parameters from global model to local model
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        learning_rate = self.conf['client']['lr']
        for i in range(len(self.conf['client']['lr_decrease_epochs'])):
            if global_epoch > self.conf['client']['lr_decrease_epochs'][i]:
                learning_rate *= 0.5
            else:
                continue

        # poison matrix
        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])
        
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(),
                                    lr=learning_rate)

        for epoch in range(self.conf['client']['epoch']):
            loss_sum = 0
            for imgs, labels in self.train_loader:
                # poison images
                for m in range(self.conf['malicious']['poison_num_per_batch']):
                    img = imgs[m].numpy()
                    for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                        img[0][pos[i][0]][pos[i][1]] = 1.0
                        img[1][pos[i][0]][pos[i][1]] = 0
                        img[2][pos[i][0]][pos[i][1]] = 0
                    labels[m] = self.conf['malicious']['poison_label']
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.local_model(imgs)

                loss = loss_function(output, labels)

                # combine malicious model and global model
                dist_loss = model.model_norm(self.local_model, global_model)
                loss = self.conf['malicious']['alpha']*loss + (1-self.conf['malicious']['alpha'])*dist_loss

                loss.backward()
                loss_sum += loss

                optimizer.step()

            print('Client {}: Epoch: {} - Loss: {} - \033[31mMalicious Client\033[0m'.format(self.id, epoch + 1, loss_sum))

        print('Learning Rate: {}'.format(learning_rate))
        difference = {} # the difference between local model and global model

        for name, param in self.local_model.state_dict().items():
            difference[name] = self.conf['malicious']['eta'] * (param - global_model.state_dict()[name]) + global_model.state_dict()[name]

        return difference