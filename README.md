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
    "honest_clients_epochs": 8,
    "weight_sign_norm_threshold": {change weight_sign_norm_threshold here},
    "server": {
        "model_type": 2,
        "epoch": 100,
        "batch_size": 32
    },
    "client": {
        "model_type": 2,
        "epoch": 5,
        "lr": 0.001,
        "batch_size": 32,
        "momentum": 0.0001
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
| 200000    | 0.7987   | 0.0288                       | 44.34%                 | 100.00%                   | log_2023-08-22_16-05-51 |
| 250000    | 0.8568   | 0.1993                       | 30.90%                 | 98.11%                    | log_2023-08-22_03-12-06 |
| 270000    | 0.7973   | 0.1161                       | 30.48%                 | 96.77%                    | log_2023-08-22_22-57-38 |
| 350000    | 0.8008   | 0.9240                       | 24.10%                 | 77.27%                    | log_2023-08-23_02-30-45 |



#### (2) Honest Clients Epochs Experiment



Parameters of this experiment:

```json
{
    "num_workers": 10,
    "subset": 3,
    "lambda": 0.3,
    "honest_clients_epochs": {change honest_clients_epochs here},
    "weight_sign_norm_threshold": 330000,
    "server": {
        "model_type": 2,
        "epoch": 100,
        "batch_size": 32
    },
    "client": {
        "model_type": 2,
        "epoch": 5,
        "lr": 0.001,
        "batch_size": 32,
        "momentum": 0.0001
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
| 4                     | 0.1000   | 1.0000                       | 98.81%                 | 48.57%                    | log_2023-08-24_22-04-09 |
| 8                     | 0.7939   | 0.2415                       | 21.54%                 | 93.33%                    | log_2023-08-26_18-26-17 |
| 16                    | 0.8069   | 0.0558                       | 17.24%                 | 100.00%                   | log_2023-08-27_00-39-00 |
| 24                    | 0.8056   | 0.1106                       | 17.22%                 | 100.00%                   | log_2023-08-24_08-12-29 |
| 32                    | 0.8032   | 0.0553                       | 13.51%                 | 100.00%                   | log_2023-08-26_21-55-37 |
