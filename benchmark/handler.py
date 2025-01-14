import re
import pandas as pd
from scipy.stats import pearsonr
import numpy as np


def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns to extract data
    shapley_pattern = r"Shapley Values: \[([-\d.,\s]+)\]"
    approx_simple_pattern = r"Approximate Banzhaf Values Simple: \[([-\d.,\s]+)\]"
    approx_hessian_pattern = r"Approximate Banzhaf Values Hessian: \[([-\d.,\s]+)\]"
    banzhaf_pattern = r"Banzhaf Values: \[([-\d.,\s]+)\]"

    # extract values
    shapley_values = [list(map(float, match.split(','))) for match in re.findall(shapley_pattern, logs)]
    approx_simple_values = [list(map(float, match.split(','))) for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [list(map(float, match.split(','))) for match in re.findall(approx_hessian_pattern, logs)]
    banzhaf_values = [list(map(float, match.split(','))) for match in re.findall(banzhaf_pattern, logs)]

    # concatenate across runs
    shapley_all = np.array([val for run in shapley_values for val in run])
    approx_simple_all = np.array([val for run in approx_simple_values for val in run])
    approx_hessian_all = np.array([val for run in approx_hessian_values for val in run])
    banzhaf_all = np.array([val for run in banzhaf_values for val in run])
    
    # create a dataframe for easier processing
    data = pd.DataFrame({
        'Shapley': shapley_all,
        'Approx_Simple': approx_simple_all,
        'Approx_Hessian': approx_hessian_all,
        'Banzhaf': banzhaf_all
    })

    # calculate and print correlations
    print(f"Pearson Correlation Coefficients for {file_path.split('/')[-1].split('.')[0].upper()}:")
    print(f"Shapley and Approx Simple: {pearsonr(data['Shapley'], data['Approx_Simple'])[0]:.4f}")
    print(f"Shapley and Approx Hessian: {pearsonr(data['Shapley'], data['Approx_Hessian'])[0]:.4f}")
    print(f"Banzhaf and Approx Simple: {pearsonr(data['Banzhaf'], data['Approx_Simple'])[0]:.4f}")
    print(f"Banzhaf and Approx Hessian: {pearsonr(data['Banzhaf'], data['Approx_Hessian'])[0]:.4f} \n")


process_log('benchmark/cifar.log')
process_log('benchmark/fmnist.log')
