import copy
import multiprocessing
import os
import time
import warnings
from collections import defaultdict
from functools import partial

import torch
from tqdm import tqdm

from options import args_parser
from update import train_client, test_inference, gradient
from utils import get_dataset, average_weights, setup_logger, common_logging, get_device, initialize_model
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
from valuation.influence import compute_influence
from valuation.shapley import compute_shapley

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device):
    """Train the global model."""
    global_weights = model.state_dict()
    abv_simple, abv_hessian, shapley_values, influence_values = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    runtimes = {'abvs': 0, 'abvh': 0, 'sv': 0, 'if': 0}
    
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        local_weights = []
        local_weights_dict = defaultdict(dict)

        start_time = time.time()
        grad = gradient(args, model, valid_dataset)
        runtimes['abvs'] += time.time() - start_time
        runtimes['abvh'] += time.time() - start_time

        # no randomization
        idxs_users = range(args.num_users)
        
        train_client_partial = partial(train_client, args=args, global_weights=copy.deepcopy(global_weights), train_dataset=train_dataset, user_groups=user_groups, epoch=epoch, device=device)
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(train_client_partial, idxs_users)
        pool.close()
        pool.join()

        for idx, w, delta in results:
            local_weights.append(copy.deepcopy(w))
            local_weights_dict[idx] = copy.deepcopy(w)
            delta_t[epoch][idx] = delta

        # compute shapley values
        start_time = time.time()
        shapley_updates = compute_shapley(args, global_weights, local_weights_dict, test_dataset)
        for k, v in shapley_updates.items():
            shapley_values[k] += v  
        runtimes['sv'] += (time.time() - start_time) * args.shapley_processes # as banzhaf is computed in a single process

        global_weights = average_weights(local_weights)
       
        # compute banzhaf values
        start_time = time.time()
        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
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
    
    # compute influence values
    start_time = time.time()
    influence_values = compute_influence(args, global_weights, train_dataset, test_dataset, user_groups)
    runtimes['if'] += time.time() - start_time

    return model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    args = args_parser()
    if args.dataset in ['fmnist', 'fmnist2']:
        logger = setup_logger(f'robustness/fmnist{args.setting}')
    elif args.dataset in ['cifar', 'resnet']:
        logger = setup_logger(f'robustness/cifar{args.setting}')
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    global_model = initialize_model(args)
    global_model.to(device)
    global_params = global_model.state_dict()

    for i in range(5):
        global_model.load_state_dict(global_params)
        logger.info(f'Run {i}')

        global_model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device)
        test_acc, test_loss = test_inference(global_model, test_dataset)

        # log results   
        common_logging(logger, args, test_acc, test_loss)
        logger.info(f'Banzhaf Values Simple: {abv_simple}')
        logger.info(f'Banzhaf Values Hessian: {abv_hessian}')
        logger.info(f'Shapley Values: {shapley_values}')
        logger.info(f'Influence Function Values: {influence_values}')
        logger.info(f'Actual Bad Clients: {actual_bad_clients}')
        logger.info(f'Runtimes: {runtimes}')