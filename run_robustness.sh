#!/bin/bash

# get processes and shapley_processes
if [ -z "$1" ]; then
    PROCESSES=8 
else
    PROCESSES=$1
fi

if [ -z "$2" ]; then
    SHAPLEY_PROCESSES=6  
else
    SHAPLEY_PROCESSES=$2
fi

echo "Using $PROCESSES processes."

python src/robustness.py --dataset cifar --setting 0 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 
python src/robustness.py --dataset cifar --setting 1 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01
python src/robustness.py --dataset cifar --setting 2 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 --badsample_prop 0.8
python src/robustness.py --dataset cifar --setting 3 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 --badsample_prop 0.8

python src/robustness.py --dataset fmnist --setting 0 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 
python src/robustness.py --dataset fmnist --setting 1 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 
python src/robustness.py --dataset fmnist --setting 2 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 --badsample_prop 0.8
python src/robustness.py --dataset fmnist --setting 3 --processes $PROCESSES --num_users 10 --local_ep 3 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES --lr 0.01 --badsample_prop 0.8

# chmod +x run_robustness.sh
# ./run_robustness.sh [processes] [shapley_processes]