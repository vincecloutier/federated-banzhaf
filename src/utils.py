import copy
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from sampling import iid, noniid, mislabeled, noisy
from models import CNNFashion, CNNCifar, ResNet9, CNNFashion2


class EarlyStopping:
    """Early stopping utility to stop training based on accuracy and/or loss conditions."""
    def __init__(self, args, patience=3, epoch_threshold=15):
        self.best_acc = -float('inf')
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.patience = patience
        self.epoch_threshold = epoch_threshold
        self.acc_threshold = 0.85 if args.dataset in ['fmnist', 'fmnist2'] else 0.80
        self.args = args

    def check(self, epoch, acc, loss):
        """Checks if early stopping should occur based on current epoch, accuracy, and loss."""
        improved_acc = acc > self.best_acc * 1.01
        improved_loss = loss < self.best_loss * 0.99

        if improved_acc or improved_loss:
            self.best_acc = max(self.best_acc, acc)
            self.best_loss = min(self.best_loss, loss)
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

            if self.args.acc_stopping == 0:
                # stop if patience exceeded and epoch is beyond threshold
                if self.no_improvement_count > self.patience and epoch > self.epoch_threshold:
                    return True
            else:
                # additionally consider accuracy threshold
                if (self.no_improvement_count > self.patience and (epoch > self.epoch_threshold or acc > self.acc_threshold)):
                    return True

        return False


class SubsetSplit(Dataset):
    """A Dataset class for creating subsets of a larger PyTorch Dataset based on a list of indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs] # indices of the dataset to include in the subset
        self.targets = np.array(self.dataset.targets)[self.idxs].copy()
        self.data = [self.dataset[idx][0].numpy() for idx in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.targets[item]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_dataset(args):
    """Returns train, validation, and test datasets and a user group (mapping user index to data indices)."""
    # define transformations for each dataset type
    t_dict = {
        'fmnist': {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
        },
        'cifar': {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
        }
    }

    # select dataset-specific configurations
    if args.dataset in ['cifar', 'resnet']:
        dataset_name = 'cifar'
        data_dir = './data/cifar/'
        dataset_class = datasets.CIFAR10
    elif args.dataset in ['fmnist', 'fmnist2']:
        dataset_name = 'fmnist'
        data_dir = './data/fmnist/'
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError("Unrecognized dataset name.")

    os.makedirs(data_dir, exist_ok=True)

    # create train and test datasets
    full_train_dataset = dataset_class(data_dir, train=True, download=True, transform=t_dict[dataset_name]['train'])
    train_dataset, valid_dataset = train_val_split(full_train_dataset, val_prop=0.1)
    test_dataset = dataset_class(data_dir, train=False, download=True, transform=t_dict[dataset_name]['test'])

    # handle different settings
    if args.setting == 0: # iid
        user_groups = iid(train_dataset, args.num_users)
        bad_clients = None
    elif args.setting == 1: # noniid
        user_groups, bad_clients = noniid(train_dataset, dataset_name, args.num_users, args.badclient_prop, args.num_categories_per_client)
    elif args.setting == 2: # mislabeled
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = mislabeled(train_dataset, dataset_name, iid_user_groups, args.badclient_prop, args.badsample_prop)
    elif args.setting == 3: # noisy
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = noisy(train_dataset, dataset_name, iid_user_groups, args.badclient_prop, args.badsample_prop)
    else:
        raise ValueError("Invalid value for --setting. Please use 0, 1, 2, or 3.")

    return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients


def train_val_split(full_train_dataset, val_prop):
    """Splits a dataset into training and validation subsets."""
    num_train = len(full_train_dataset)
    split = int(np.floor(val_prop * num_train))
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubsetSplit(full_train_dataset, train_idx), SubsetSplit(full_train_dataset, valid_idx)


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    del w
    torch.cuda.empty_cache()

    return w_avg


def initialize_model(args):
    """Initializes a model based on the dataset argument."""
    model_dict = {
        'fmnist': CNNFashion,
        'fmnist2': CNNFashion2,
        'cifar': CNNCifar,
        'resnet': ResNet9,
    }

    if args.dataset in model_dict:
        return model_dict[args.dataset](args=args)
    else:
        raise ValueError('Error: unrecognized dataset for model initialization.')


def get_device():
    """Returns the appropriate torch.device object depending on CUDA availability."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 2) -> list[int]:
    """Identifies indices (e.g., clients) that have a Banzhaf value below a threshold or negative."""
    if not approx_banzhaf_values:
        return []
    avg_banzhaf = np.mean(list(approx_banzhaf_values.values()))
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < avg_banzhaf / threshold or banzhaf < 0]
    return bad_idxs


def measure_accuracy(targets, predictions):
    """Calculates the accuracy metric (TP + TN) / (TP + TN + FP + FN)."""
    if targets is None or predictions is None:
        return 0.0
    if len(targets) == 0 and len(predictions) == 0:
        return 1.0
        
    targets, predictions = set(targets), set(predictions)
    TP = len(predictions & targets)
    FP = len(predictions - targets)
    FN = len(targets - predictions)

    universe = targets | predictions
    TN = len(universe - (targets | predictions))

    return (TP + TN) / (TP + TN + FP + FN)


def setup_logger(strategy_name: str) -> logging.Logger:
    """Sets up a logger for a given strategy."""
    logger = logging.getLogger(strategy_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{strategy_name}.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)

    # ensure we don't add multiple handlers if logger is re-initialized
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def common_logging(logger, args, test_acc, test_loss):
    """Log the common experiment details."""
    bad_clients = args.badclient_prop * args.num_users

    if args.setting == 0:
        setting_str = "IID"
    elif args.setting == 1:
        setting_str = f"Non IID with {bad_clients} Bad Clients and {args.num_categories_per_client} Categories Per Bad Client"
    elif args.setting == 2:
        setting_str = f"Mislabeled with {bad_clients} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
    elif args.setting == 3:
        setting_str = f"Noisy with {bad_clients} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"

    logger.info(f"Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}, Number Of Clients: {args.num_users}")
    logger.info(f"Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}, Learning Rate: {args.lr}")
    logger.info(f"Test Accuracy Before Retraining: {test_acc * 100:.2f}, Test Loss Before Retraining: {test_loss}")