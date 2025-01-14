#!/bin/bash

# Default values
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES="$1"
fi

for i in {1..3}
do  
    echo "Run $i"
    for setting in {0..3}
    do
        python src/benchmark.py --dataset cifar --setting $setting --processes $PROCESSES --local_ep 3
        python src/benchmark.py --dataset fmnist --setting $setting --processes $PROCESSES --local_ep 3 --acc_stopping 0
    done
done

echo "All runs completed."

# chmod +x run_benchmark.sh
# ./run_benchmark.sh [processes]