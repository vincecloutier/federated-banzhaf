import copy
import multiprocessing
import os
import time
import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from options import args_parser
from update import train_client, test_inference, gradient
from utils import get_dataset, average_weights, setup_logger, common_logging, get_device, identify_bad_idxs, measure_accuracy, initialize_model, EarlyStopping
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device, bad_clients=None):
    initial_start_time = time.time()
    global_weights = model.state_dict()
    abv_simple, abv_hessian = defaultdict(float), defaultdict(float)
    delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})
    
    runtimes = {'abvs': 0, 'abvh': 0, 'total': 0}
    early_stopping = EarlyStopping(args)

    for epoch in tqdm(range(args.epochs)):
        local_weights = []

        start_time = time.time()
        grad = gradient(args, model, valid_dataset)
        runtimes['abvs'] += time.time() - start_time
        runtimes['abvh'] += time.time() - start_time

        if bad_clients is not None:
            good_clients = [i for i in range(args.num_users) if i not in bad_clients]
            if len(good_clients) == 0:
                good_clients = range(args.num_users)
            m = max(int(args.frac * len(good_clients)), 1)
            idxs_users = np.random.choice(good_clients, m, replace=False)
        else:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
       
        train_client_partial = partial(train_client, args=args, global_weights=copy.deepcopy(global_weights), train_dataset=train_dataset, user_groups=user_groups, epoch=epoch)
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(train_client_partial, idxs_users)
        pool.close()
        pool.join()

        for idx, w, delta in results:
            local_weights.append(copy.deepcopy(w))
            delta_t[epoch][idx] = delta

        global_weights = average_weights(local_weights)

        # compute banzhaf values
        start_time = time.time()
        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
                    if delta_g[idx][key].dtype != G_t_minus_i[key].dtype or delta_g[idx][key].dtype != G_t[key].dtype:
                        raise ValueError(f"delta_g[{idx}][{key}].dtype: {delta_g[idx][key].dtype}, G_t_minus_i[{key}].dtype: {G_t_minus_i[key].dtype}, G_t[{key}].dtype: {G_t[key].dtype}")
                    delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
            t_time = time.time() - start_time
            runtimes['abvh'] += t_time
            runtimes['abvs'] += t_time
            start_time = time.time()
            abv_hessian[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=True)
            runtimes['abvh'] += time.time() - start_time
            start_time = time.time()
            abv_simple[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=False)
            runtimes['abvs'] += time.time() - start_time

        model.load_state_dict(global_weights)

        with torch.no_grad():
            acc, loss = test_inference(model, test_dataset)
        
        if early_stopping.check(epoch, acc, loss):
            print(f'Convergence Reached At Round {epoch + 1}')
            break

        print(f'Epoch {epoch+1}/{args.epochs} - Test Accuracy: {acc}, Test Loss: {loss}, Runtimes: {runtimes}s')
        
    runtimes['total'] = time.time() - initial_start_time
    return model, abv_simple, abv_hessian, runtimes


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    args = args_parser()
    if args.dataset in ['fmnist', 'fmnist2']:
        logger = setup_logger(f'retraining/fmnist{args.setting}')
    elif args.dataset in ['cifar', 'resnet']:
        logger = setup_logger(f'retraining/cifar{args.setting}')
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model, abv_simple, abv_hessian, runtimes = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(global_model, test_dataset)

    # predict bad clients and measure accuracy
    predicted_bad_abvs = identify_bad_idxs(abv_simple)
    predicted_bad_abvh = identify_bad_idxs(abv_hessian)
    bad_client_accuracy_abvs = measure_accuracy(actual_bad_clients, predicted_bad_abvs)
    bad_client_accuracy_abvh = measure_accuracy(actual_bad_clients, predicted_bad_abvh)
    
    print(f'Bad Client Accuracy: {bad_client_accuracy_abvs}')

    # retrain the model w/o bad clients 
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    retrained_model, _, _, retrained_runtimes = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, predicted_bad_abvh)
    retrain_test_acc, retrain_test_loss = test_inference(retrained_model, test_dataset)

    # log results
    common_logging(logger, args, test_acc, test_loss)
    logger.info(f'Test Accuracy After Retraining: {100*retrain_test_acc}, Test Loss After Retraining: {retrain_test_loss}')