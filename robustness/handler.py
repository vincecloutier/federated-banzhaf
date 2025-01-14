import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    shapley_pattern = r"Shapley Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    influence_pattern = r"Influence Function Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    runtime_pattern = r"Runtimes: \{(.*?)\}"

    # extract values
    approx_simple_values = [eval(f"{{{match}}}") for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [eval(f"{{{match}}}") for match in re.findall(approx_hessian_pattern, logs)]
    shapley_values = [eval(f"{{{match}}}") for match in re.findall(shapley_pattern, logs)]
    influence_values = [eval(f"{{{match}}}") for match in re.findall(influence_pattern, logs)]
    runtimes = [eval(f"{{{match}}}") for match in re.findall(runtime_pattern, logs)]

    return approx_simple_values, approx_hessian_values, shapley_values, influence_values, runtimes
    

def compute_rank_stability(runs):
    df = pd.DataFrame(runs)
    df = df.sort_index(axis=1)
    df = df.fillna(np.nan)
    ranked_df = df.rank(axis=1, method='average', ascending=False)
    ranked_df_T = ranked_df.transpose()
    correlations = spearmanr(ranked_df_T, nan_policy='omit')[0]
    print(correlations.mean())
    return correlations.mean()


def compute_rank_stability_med(runs):
    df = pd.DataFrame(runs)
    df = df.sort_index(axis=1)
    df = df.fillna(np.nan)
    ranked_df = df.rank(axis=1, method='average', ascending=False)
    median_ranks = ranked_df.median(axis=0)
    corrs = []
    for i in range(len(ranked_df)):
        run_ranks = ranked_df.iloc[i, :]
        corr, _ = spearmanr(run_ranks, median_ranks, nan_policy='omit')
        corrs.append(corr)
    print(np.mean(corrs))
    return np.mean(corrs)


def process_and_graph_logs(log_files, plot=False):
    # lists to hold rank stability and runtimes per setting
    corr_metrics = {'FBVS': [], 'FBVH': [], 'FSV': [], 'Influence': []}
    runtime_metrics = {'FBVS': [], 'FBVH': [], 'FSV': [], 'Influence': []}

    # process each log file (each representing a setting)
    for log_file in log_files:
        print(log_file)
        approx_simple, approx_hessian, shapley, influence, runtimes = process_log(log_file)
        corr_metrics['FBVS'].append(compute_rank_stability_med(approx_simple))
        corr_metrics['FBVH'].append(compute_rank_stability_med(approx_hessian))
        corr_metrics['FSV'].append(compute_rank_stability_med(shapley))
        corr_metrics['Influence'].append(compute_rank_stability_med(influence))
        for run in runtimes:
            runtime_metrics["FBVS"].append(run["abvs"])
            runtime_metrics["FBVH"].append(run["abvh"])
            runtime_metrics["FSV"].append(run["sv"])
            runtime_metrics["Influence"].append(run["if"])
    
    # compute summary statistics across settings
    methods = ["FBVS", "FBVH", "Influence", "FSV"]
    avg_corrs = [np.mean(corr_metrics[m]) for m in methods]
    avg_runtimes = [np.mean(runtime_metrics[m]) for m in methods]

    if plot:
        # generate plots
        dataset = re.search(r'/(.+)\d', log_files[0]).group(1)
        colors = plt.get_cmap('tab10').colors
        colors = [colors[0], colors[2], colors[1], colors[3]]

        plt.figure(figsize=(8, 4.5), layout="constrained")
        plt.subplot(1, 2, 1)
        
        plt.bar(methods, avg_corrs, capsize=5, color=colors)
        for i, mean in enumerate(avg_corrs):
            plt.text(i, mean, f'{mean:.2f}', ha='center', va='bottom')
        plt.title("Average Spearman Rank Correlation")
        plt.xlabel("Data Valuation Method")
        plt.ylim(0, 1)
        plt.ylabel("Correlation")

        plt.subplot(1, 2, 2)
        plt.bar(methods, avg_runtimes, color=colors)
        for i, v in enumerate(avg_runtimes):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.yscale('log')
        plt.title("Average Runtime")
        plt.xlabel("Data Valuation Method")
        plt.ylabel("Runtime (s)")
        plt.gca().yaxis.set_label_position('right')
        plt.gca().yaxis.tick_right()

        plt.savefig(f"robustness/graphs/robustness_{dataset}.png", dpi=150, bbox_inches='tight')


process_and_graph_logs(['robustness/cifar0.log', 'robustness/cifar1.log', 'robustness/cifar2.log', 'robustness/cifar3.log'], plot=True)
process_and_graph_logs(['robustness/fmnist0.log', 'robustness/fmnist1.log', 'robustness/fmnist2.log', 'robustness/fmnist3.log'], plot=True)