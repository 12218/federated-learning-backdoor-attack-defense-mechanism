## Defending Against Federated Learning Backdoor Attacks: Defense Strategies and  Performance Evaluation

### 01 Overview

This repository is the code implementation of "Defending Against Federated Learning Backdoor Attacks: Defense Strategies and  Performance Evaluation".

I conducted defense experiments against backdoor attacks in Federated Learning using the Cifar-10 dataset and the ResNet18 model.

### 02 Run the Codes

Training the Federated Learning model:

```bash
python main.py -c ./config/config.json
```

Test the accuracy and the accuracy on poisoned dataset:

```bash
python model_test.py -c ./config/config.json
```

The configuration file is in `./config/config.json` .

Here is the description of some parameters in `config.json` :

```json
{
    "num_workers": 10, // 10 clients in total
    "subset": 3, // select 3 clients every epoch
    "lambda": 0.3,
    "honest_clients_epochs": 8, // honest clients epochs
    "weight_sign_norm_threshold": 330000, // weight sign norm threshold
    "server": {
        "model_type": 2, // model type: 1 - a simple CNN; 2 - ResNet18
        "epoch": 100, // server runs 100 epochs of training
        "batch_size": 32
    },
    "client": {
        "model_type": 2, // model type: 1 - a simple CNN; 2 - ResNet18
        "epoch": 5, // clients run 5 epochs in every global epoch
        "lr": 0.001, // learning rate
        "batch_size": 32,
        "momentum": 0.0001
    },
    "malicious": {
        "poison_num_per_batch": 4, // 4 poisoned images in each batch
        "poison_label": 2, // the poisoned images' label is 2
        "alpha": 1.0, // backdoor attack parameter
        "eta": 2 // backdoor attack parameter
    }
}
```

| Term                       | Description                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| honest_clients_epochs      | The number of epochs that only honest clients are involved in at the beginning of training.                        |
| weight_sign_norm_threshold | The threshold for **weight_sign_norm** to discern whether the weights of a model originate from malicious clients. |
| epoch                      | The number of iterations that the server distributes the global model to the clients.                              |

### 03 Experiments

Launch Tensorboard for experiments:

```bash
tensorboard --host=0.0.0.0 --port=6007 --logdir=logs/{log file}
```

#### (1) Weight Sign Norm Threshold Experiment

Parameters of this experiment:

```json
{
    "num_workers": 10,
    "subset": 3,
    "lambda": 0.3,
    "honest_clients_epochs": 16,
    "weight_sign_norm_threshold": {change weight_sign_norm_threshold here},
    "server": {
        "model_type": 3,
        "epoch": 200,
        "batch_size": 4
    },
    "client": {
        "model_type": 3,
        "epoch": 3,
        "lr": 0.001,
        "batch_size": 32,
        "lr_decrease_epochs": [60, 100, 160, 200],
        "momentum": 0.9
    },
    "malicious": {
        "poison_num_per_batch": 4,
        "poison_label": 2,
        "alpha": 1.0,
        "eta": 2
    }
}
```

| Threshold | Accuracy | Accuracy on Poisoned Dataset | Ignored Honest Clients | Ignored Malicious Clients | Log File                |
| --------- | -------- | ---------------------------- | ---------------------- | ------------------------- | ----------------------- |
| 270000    | 0.9240   | 0.1576                       | 5.84%                  | 100.00%                   | log_2023-09-10_18-54-37 |
| 300000    | 0.9304   | 0.2832                       | 0.81%                  | 100.00%                   | log_2023-09-10_22-19-33 |
| 330000    | 0.9264   | 0.1592                       | 0.02%                  | 100.00%                   | log_2023-09-11_02-05-28 |
| 400000    | 0.9288   | 0.1072                       | 0.00%                  | 100.00%                   | log_2023-09-28_16-57-38 |
| 450000    | 0.9208   | 0.5992                       | 0.00%                  | 94.23%                    | log_2023-09-28_20-17-29 |

#### (2) Honest Clients Epochs Experiment

Parameters of this experiment:

```json
{
    "num_workers": 10,
    "subset": 3,
    "lambda": 0.3,
    "honest_clients_epochs": {change honest_clients_epochs here},
    "weight_sign_norm_threshold": 400000,
    "server": {
        "model_type": 3,
        "epoch": 200,
        "batch_size": 4
    },
    "client": {
        "model_type": 3,
        "epoch": 3,
        "lr": 0.001,
        "batch_size": 32,
        "lr_decrease_epochs": [60, 100, 160, 200],
        "momentum": 0.9
    },
    "malicious": {
        "poison_num_per_batch": 4,
        "poison_label": 2,
        "alpha": 1.0,
        "eta": 2
    }
}
```

| Honest Clients Epochs | Accuracy | Accuracy on Poisoned Dataset | Ignored Honest Clients | Ignored Malicious Clients | Log File                |
| --------------------- | -------- | ---------------------------- | ---------------------- | ------------------------- | ----------------------- |
| 4                     | 0.9208   | 0.4104                       | 0.00%                  | 95.45%                    | log_2023-09-29_16-14-52 |
| 8                     | 0.9184   | 0.4120                       | 0.00%                  | 96.00%                    | log_2023-09-29_10-06-01 |
| 16                    | 0.9288   | 0.1072                       | 0.00%                  | 100.00%                   | log_2023-09-28_16-57-38 |
