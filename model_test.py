import argparse, json
import torch
import data, server

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('-c', '--conf', '--configuration', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = json.load(file)

    # load dataset
    train_set, test_set = data.get_data(dir='./datasets/', conf=conf, return_type='dataset')
    train_loader, test_loader = data.get_data(dir='./datasets/', conf=conf, return_type='dataloader')

    server = server.Server(conf=conf, test_dataset=test_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # server.global_model = torch.load(f='./saved_models/model-2023-07-25_01-34-06_final.pth')
    # server.global_model = torch.load(f='./saved_models/model-2023-07-25_10-55-14.pth')
    # server.global_model = torch.load(f='./saved_models/model-2023-07-25_13-07-07.pth')
    server.global_model = torch.load(f='./saved_models/model-2023-09-09_10-27-00_accuracy-0.9240.pth')

    pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])

    for imgs, labels in test_loader:
        poisoned_labels = labels.clone()
        for m in range(imgs.shape[0]):
            img = imgs[m].numpy()
            for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                img[0][pos[i][0]][pos[i][1]] = 1.0
                img[1][pos[i][0]][pos[i][1]] = 0
                img[2][pos[i][0]][pos[i][1]] = 0
            poisoned_labels[m] = conf['malicious']['poison_label']

        imgs, labels = imgs.to(device), labels.to(device)

        output = server.global_model(imgs)
        prediction = torch.max(output, 1)

        print("Prediction:\n{}".format(prediction.indices))
        print("Real Labels:\n{}".format(labels))

        break

    correct_count = 0
    all_count = 0
    for imgs, labels in test_loader:
        poisoned_labels = labels.clone()
        for m in range(imgs.shape[0]):
            img = imgs[m].numpy()
            for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                img[0][pos[i][0]][pos[i][1]] = 1.0
                img[1][pos[i][0]][pos[i][1]] = 0
                img[2][pos[i][0]][pos[i][1]] = 0
            poisoned_labels[m] = conf['malicious']['poison_label']

        imgs, labels = imgs.to(device), labels.to(device)

        output = server.global_model(imgs)
        prediction = torch.max(output, 1)

        # print("Prediction:\n{}".format(prediction.indices))
        # print("Real Labels:\n{}".format(labels))
        correct_count += len([x for x in prediction.indices if x == 2])
        all_count += len(prediction.indices)

    print("Prediction Accuracy on Poisoned Data: {}".format(correct_count / all_count))