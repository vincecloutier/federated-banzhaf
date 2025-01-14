# Federated Banzhaf Value & FBV-Debug Experiments
This repository contains the experiments accompanying the paper [Robust Client-level Contribution Assessment in Horizontal Federated Learning](paper.pdf). This project was undertaken under the supervision of Dr. Zhuan Shi and Professor Boi Faltings at EPFL's Artificial Intelligence Laboratory.


## Installation

1. **Clone the Repository**  
    ```bash
    git clone https://github.com/vincecloutier/federated-banzhaf.git
    ```
2. **Install Dependencies**  
    ```bash
    cd federated-banzhaf
    pip install requirements.txt 
    # for the robustness comparison to an influence based scheme
    git clone https://github.com/alstonlo/torch-influence
    cd torch-influence
    pip install -e .
    cd ..
    ```

## Usage
The following experiments are all implemented in the `src` directory, but can be run from the root directory with the following commands. Note that in all cases, you need to run the corresponding processing scripts in the `benchmark`, `robustness`, and `retraining` directories. They are explained in greater detail in the paper.

### Benchmark
This experiment establishes a ground truth between the FBV and the Shapley value by measuring the Pearson correlation between the two. It is implemented in `src/benchmark.py`, and we process the results in `benchmark/handler.py`. To run the experiment, use the following commands:
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh [processes]
```
where `processes` is the number of processes to use for the multiprocessing pool.

### Robustness
This experiment measures the rank stability of the FBV compared to the Federated Shapley Value and an Influence Function based scheme. It is implemented in `src/robustness.py`, and we process the results in `robustness/handler.py`. To run the experiment, use the following commands:
```bash
chmod +x run_robustness.sh
./run_robustness.sh [processes] [shapley_processes]
```
where `processes` is the number of processes to use for the multiprocessing pool, and `shapley_processes` is the number of processes to use for the Shapley value calculation.

### Retraining
This experiment measures the performance of the FBV-Debug across a wide range of data settings. It is implemented in `src/retraining.py`, and we process the results in `retraining/handler.py`. To run the experiment, use the following commands:
```bash
chmod +x run_retraining.sh
./run_retraining.sh [processes]
```
where `processes` is the number of processes to use for the multiprocessing pool.