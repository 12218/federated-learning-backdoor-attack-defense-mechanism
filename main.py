from datetime import datetime
import time
import random
import torch
import argparse, json
import data
from server import Server
from client import Client
from torch.utils.tensorboard.writer import SummaryWriter
from estimate_finishing_time import EstimateTime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('-c', '--conf', '--configuration', dest='conf')
    args = parser.parse_args()

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    writer = SummaryWriter(log_dir='./logs/log_{}'.format(time_str))

    with open(args.conf, 'r') as file:
        conf = json.load(file)

    # a tool to estimate the finishing time
    time_estimate = EstimateTime(epoch=conf['server']['epoch'], start_time=time.time())

    # load dataset
    train_set, test_set = data.get_data(dir='./datasets/', conf=conf, return_type='dataset')

    # instantiate server and clients
    server = Server(conf=conf, test_dataset=test_set)
    clients = []

    for i in range(conf['num_workers']):
        clients.append(Client(conf=conf, train_dataset=train_set, id=i))

    previous_weight_sign_dict = {}

    honest_client_count = 0
    ignored_honest_client_count = 0
    malicious_client_count = 0
    ignored_malicious_client_count = 0

    # training
    for epoch in range(conf['server']['epoch']):
        participants = random.sample(clients, conf['subset'])

        weight_accumulator = {}
        subset_diff_list = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params) # initialise accumulator for each layer

        for p in participants:
            if p.id == 1 and epoch >= conf['honest_clients_epochs']: # malicious client; only honest clients in first 10 epochs
                malicious_client_count += 1
                difference = p.local_train_malicious(server.global_model, epoch)
            else:
                honest_client_count += 1
                difference = p.local_train(server.global_model, epoch)

            if epoch < conf['honest_clients_epochs']:
                for name, params in server.global_model.state_dict().items():
                    weight_sign_sum = torch.sign(difference[name]).sum() # weight_sign_sum will be compared to norm threshold
                    previous_weight_sign_dict[name] = weight_sign_sum
                    weight_accumulator[name].add_(difference[name])
            else:
                weight_sign_dict = {}
                weight_norm = 0
                for name, params in server.global_model.state_dict().items():
                        weight_sign_sum = torch.sign(difference[name]).sum() # weight_sign_sum will be compared to norm threshold
                        weight_sign_dict[name] = weight_sign_sum

                for name, params in previous_weight_sign_dict.items():
                    weight_norm += torch.norm(previous_weight_sign_dict[name].float() - weight_sign_dict[name].float())
                    
                if weight_norm <= conf['weight_sign_norm_threshold']: # if the the weight norm of this client can be considered as a honest client, do the accumulation
                    print('Weight Sign Norm: {}'.format(weight_norm))
                    for name, params in server.global_model.state_dict().items():
                        weight_accumulator[name].add_(difference[name])

                    # update the previous_weight_sign_dict
                    for name, params in previous_weight_sign_dict.items():
                        previous_weight_sign_dict[name] = (previous_weight_sign_dict[name] + weight_sign_dict[name]) / 2
                else:
                    if p.id == 1:
                        ignored_malicious_client_count += 1
                    else:
                        ignored_honest_client_count += 1
                    print('Weight Sign Norm: {} - \033[31mIgnored\033[0m'.format(weight_norm))

            # subset_diff_list.append(difference) # append difference of each client into a list
            # subset_diff_list[p.id] = difference

        # if epoch < conf['honest_clients_epochs']:
        #     for pid, diff in subset_diff_list.items():
        #         for name, params in server.global_model.state_dict().items():
        #             weight_sign_sum = torch.sign(diff[name]).sum() # weight_sign_sum will be compared to norm threshold
        #             previous_weight_sign_dict[name] = weight_sign_sum
        #             weight_accumulator[name].add_(diff[name])

        # else:
        #     weight_sign_dict = {}
        #     weight_norm = 0
        #     for pid, diff in subset_diff_list.items():
        #         for name, params in server.global_model.state_dict().items():
        #             weight_sign_sum = torch.sign(diff[name]).sum() # weight_sign_sum will be compared to norm threshold
        #             weight_sign_dict[name] = weight_sign_sum

        #         for name, params in previous_weight_sign_dict.items():
        #             weight_norm += torch.norm(previous_weight_sign_dict[name].float() - weight_sign_dict[name].float())
                    
        #         if weight_norm <= conf['weight_sign_norm_threshold']: # if the the weight norm of this client can be considered as a honest client, do the accumulation
        #             print('Weight Sign Norm: {}'.format(weight_norm))
        #             for name, params in server.global_model.state_dict().items():
        #                 weight_accumulator[name].add_(diff[name])

        #             # update the previous_weight_sign_dict
        #             for name, params in previous_weight_sign_dict.items():
        #                 previous_weight_sign_dict[name] = (previous_weight_sign_dict[name] + weight_sign_dict[name]) / 2
        #         else:
        #             if pid == 1:
        #                 ignored_malicious_client_count += 1
        #             else:
        #                 ignored_honest_client_count += 1
        #             print('Weight Sign Norm: {} - Ignored'.format(weight_norm))

        server.model_aggregate(weight_accumulator=weight_accumulator)

        accuracy, loss = server.model_evaluation()
        poisoned_accuracy, poisoned_loss = server.model_poisoned_dataset_evaluation()

        # writer.add_scalar(tag='Loss', scalar_value=loss, global_step=epoch + 1)
        # writer.add_scalar(tag='Accuracy', scalar_value=accuracy, global_step=epoch + 1)

        writer.add_scalars('Loss', {'Loss': loss, 'Loss on Poisoned Data': poisoned_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Accuracy': accuracy, 'Accuracy on Poisoned Data': poisoned_accuracy}, epoch + 1)
        writer.add_scalars('Honest', {'Honest Clients': honest_client_count, 'Ignored Honest Client': ignored_honest_client_count}, epoch + 1)
        writer.add_scalars('Malicious', {'Malicious Clients': malicious_client_count, 'Ignored Malicious Client': ignored_malicious_client_count}, epoch + 1)

        # print('-> Server: Epoch: {} - Accuracy: {:.4f} - Loss: {:.6f}'.format(epoch, accuracy, loss))
        print('-> Server: Epoch: {} - Accuracy: {:.4f} - Accuracy on Poisoned Data: {:.4f} - Loss: {:.6f}'.format(epoch, accuracy, poisoned_accuracy, loss))
        print('Estimate Finishing Time: {}\n\n'.format(time_estimate.estimate(now_epoch=epoch)))

if malicious_client_count != 0: # avoid division by zero
    print("Ignored Honest Clients: {:.2f}%, Ignored Malicious Clients: {:.2f}%\n\n".format(ignored_honest_client_count / (honest_client_count - conf['honest_clients_epochs'] * 3) * 100, ignored_malicious_client_count / malicious_client_count * 100))
torch.save(server.global_model, './saved_models/model-{}_accuracy-{:.4f}.pth'.format(time_str, accuracy))

# from email_sending import Email

# email = Email()
# email.Send(['2402767956@qq.com'], 'Training Finished', 'Accuracy on test dataset:{}'.format(accuracy))

from wx_sending import WeChatSending

wechat = WeChatSending()
wechat.send(title='Training Result', name='Acc: {}, P_Acc: {}'.format(accuracy, poisoned_accuracy), content='Test')