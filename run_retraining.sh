#!/bin/bash

# get processes argument
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES="$1"
fi

# install bc if not installed
if ! command -v bc &> /dev/null; then
    echo "'bc' is not installed. Attempting to install it..."
    apt-get update && apt-get install -y bc
else
    echo "'bc' is already installed."
fi

# loop to run each command nine times
for i in {1..4}
do
    # calculate badclient_prop for this iteration
    BAD_CLIENT_PROP=$(echo "scale=1; 0.2 * $i" | bc)

    echo "Run $i with bad_client_prop=$BAD_CLIENT_PROP"

    # cifar 
    python src/retraining.py --dataset resnet --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 4
    python src/retraining.py --dataset resnet --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.6
    python src/retraining.py --dataset resnet --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.6

    python src/retraining.py --dataset resnet --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 6
    python src/retraining.py --dataset resnet --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.4 
    python src/retraining.py --dataset resnet --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.4

    python src/retraining.py --dataset resnet --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 8
    python src/retraining.py --dataset resnet --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.2
    python src/retraining.py --dataset resnet --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.2

    # fmnist
    python src/retraining.py --dataset fmnist --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 4 --lr 1e-2
    python src/retraining.py --dataset fmnist --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.6 --lr 1e-1
    python src/retraining.py --dataset fmnist2 --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 3 --badsample_prop 0.9 --method abvh --lr 5e-2

    python src/retraining.py --dataset fmnist --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 6 --lr 1e-2
    python src/retraining.py --dataset fmnist --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.4 --lr 1e-1
    python src/retraining.py --dataset fmnist2 --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 3 --badsample_prop 0.8 --method abvh --lr 1e-1

    python src/retraining.py --dataset fmnist --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --num_categories_per_client 8 --lr 1e-2
    python src/retraining.py --dataset fmnist --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 10 --badsample_prop 0.2 --lr 1e-1
    python src/retraining.py --dataset fmnist2 --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep 3 --badsample_prop 0.7 --method abvh --lr 1e-1

done

echo "All runs completed."

# chmod +x run_retraining.sh
# ./run_retraining.sh [processes]