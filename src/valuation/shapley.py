import itertools
import multiprocessing
from collections import defaultdict
from math import factorial as fact

import numpy as np
import torch

from update import test_inference
from utils import average_weights, initialize_model, get_device

def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values using permutation sampling from Federated Shapley Value Paper."""
    device = get_device()

    # initialize model and compute base accuracy
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)

    with torch.no_grad():
        base_score = test_inference(model, test_dataset)[1]

    client_keys = list(client_weights.keys())
    t = int((2 / 0.5**2) * np.log(2 * len(client_keys) / 0.5))

    shapley_updates = defaultdict(float)
    if t < fact(len(client_keys)):
        args_list = [(None, global_weights, client_weights, test_dataset, base_score, device, args) for _ in range(t)]
    else:
        t = fact(len(client_keys))
        args_list = [(perm, global_weights, client_weights, test_dataset, base_score, device, args) for perm in itertools.permutations(client_keys)]

    pool = multiprocessing.Pool(processes=args.shapley_processes)
    results = pool.map(compute_shapley_for_permutation, args_list)
    pool.close()
    pool.join()

    for result in results:
        for k, v in result.items():
            shapley_updates[k] += v

    del model, results
    torch.cuda.empty_cache()

    # average the values over all permutations
    shapley_updates = {k: v / t for k, v in shapley_updates.items()}
    return shapley_updates


def compute_shapley_for_permutation(args):
    """Compute Shapley values for a given permutation."""
    (perm, global_weights, client_weights, test_dataset, base_score, device, args_model) = args

    model = initialize_model(args_model)
    model.load_state_dict(global_weights)
    model.to(device)

    shapley_updates_local = defaultdict(float)

    if perm is None:
        perm = np.random.permutation(list(client_weights.keys()))

    prev_score = base_score
    current_weights = []

    for i in perm:
        current_weights.append(client_weights[i])
        avg_weights = average_weights(current_weights)
        model.load_state_dict(avg_weights)

        with torch.no_grad():
            curr_score = test_inference(model, test_dataset)[1]

        shapley_updates_local[i] += prev_score - curr_score
        prev_score = curr_score

    del model, current_weights
    torch.cuda.empty_cache()

    return shapley_updates_local