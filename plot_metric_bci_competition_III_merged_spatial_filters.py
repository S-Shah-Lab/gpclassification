"""
Plot the results coming from `run_class_bci_competition_III_merged_spatial_filters.py`
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ---- Input root directory ----
#root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_merged/spatialFilter"
root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_merged/spatialFilter_CSP"

result_dict = {}

k_fold = None
n_subs = 5
n_train = None
kernel_type = None
logged_flag = None

for nf_name in sorted(os.listdir(root)):
    nf_path = os.path.join(root, nf_name) # create path to here
    if not os.path.isdir(nf_path): continue
    print(f"  {nf_name}")

    nf_dummy = int(nf_name.split('_')[1])
    result_dict[nf_dummy] = {} # a dict for each num of spatial filters

    # Define buckets to store info
    nlml = [] # float
    acc_train = [] # float
    acc_test = [] # float
    brier_train = [] # float
    brier_test = [] # float

    for run_name in sorted(os.listdir(nf_path)):
        run_path = os.path.join(nf_path, run_name) # create path to here
        if not os.path.isdir(run_path): continue
        print(f"     {run_name}")

        metrics_path = os.path.join(run_path, "run_log.json")
        if os.path.isfile(metrics_path):
            # Load run_log.json and collect values
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
   
            n_train = data['meta']['N_train']
            kernel_type = data['meta']['kernel_type']
            logged_flag = data['meta']['logged_flag']

            best_iter = data["meta"]["best_iter"]

            # Find the log dict with matching "step"
            match = None
            for d in data["logs"]:
                if d["step"] == best_iter:
                    match = d
                    break

            # Store into buckets
            nlml.append(        match["nlml"]        )
            acc_train.append(   match["acc_train"]   )
            acc_test.append(    match["acc_test"]    )
            brier_train.append( match["brier_train"] )
            brier_test.append(  match["brier_test"]  )

    k_fold = len(nlml)

    result_dict[nf_dummy]['nlml'       ] = nlml
    result_dict[nf_dummy]['acc_train'  ] = acc_train
    result_dict[nf_dummy]['acc_test'   ] = acc_test
    result_dict[nf_dummy]['brier_train'] = brier_train
    result_dict[nf_dummy]['brier_test' ] = brier_test


def plot_nfs_runs(bucket: dict, title: str = None) -> None:
    """
    A dictionary with all info related to a given train / test split is given in input as `bucket`
    bucket.keys() are the number of spatial filters

    {nf0: {"nlml": [...],
           "acc_train": [...],
           "acc_test": [...],
           "brier_train": [...],
           "brier_test": [...]
        },
    }
    """
    labels = sorted(bucket.keys())

    nlmls = []
    acc_trains = []
    acc_tests = []
    brier_trains = []
    brier_tests = []

    for key in labels:

        nlmls.append(       bucket[key]['nlml']        )
        acc_trains.append(  bucket[key]['acc_train']  )
        acc_tests.append(   bucket[key]['acc_test']   )
        brier_trains.append(bucket[key]['brier_train'])
        brier_tests.append( bucket[key]['brier_test'] )

    medianprops_train    = dict(linestyle='-', linewidth=2.5, color='blue')
    medianprops_test     = dict(linestyle='-', linewidth=2.5, color='orange')
    meanpointprops_train = dict(marker='o', markerfacecolor='blue', markersize=7, markeredgecolor='none')
    meanpointprops_test  = dict(marker='^', markerfacecolor='orange', markersize=7, markeredgecolor='none')
    flierprops_train     = dict(marker='o', markerfacecolor='none', markersize=4, markeredgecolor='blue')
    flierprops_test      = dict(marker='^', markerfacecolor='none', markersize=4, markeredgecolor='orange')

    handles = {'train median': Line2D([], [], **medianprops_train), 
               'test median' : Line2D([], [], **medianprops_test), 
               'train mean'  : Line2D([], [], **meanpointprops_train), 
               'test mean'   : Line2D([], [], **meanpointprops_test), 
    }

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(f'{title}')
    ax = ax.ravel()
    # NLML scatter per num of spatial filters
    ax[0].boxplot(nlmls, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[0].set_xlabel("Number of spatial filters")
    ax[0].set_ylabel("NLML")    
    handle_labels = ['train median', 'train mean']
    ax[0].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')
    # accuracy scatter per num of spatial filters
    ax[1].boxplot(acc_trains, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[1].boxplot(acc_tests,  tick_labels=[''] * len(labels), flierprops=flierprops_test, medianprops=medianprops_test,  meanprops=meanpointprops_test, showmeans=True)
    ax[1].set_xlabel("Number of spatial filters")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[1].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')
    # brier's scatterrper num of spatial filters
    ax[2].boxplot(brier_trains, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[2].boxplot(brier_tests,  tick_labels=[''] * len(labels), flierprops=flierprops_test, medianprops=medianprops_test,  meanprops=meanpointprops_test, showmeans=True)
    ax[2].set_xlabel("Number of spatial filters")
    ax[2].set_ylabel("Brier score") 
    ax[2].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[2].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')

    plt.tight_layout()
    plt.show()



first_line = rf'N_train = {n_train} ({k_fold} folds x {n_subs} subjects)'

if kernel_type == 'RBF':
    second_line = 'RBF covariance function'
else:
    second_line = 'linear covariance function'

if logged_flag:
    third_line = 'log-variance space'
else:
    third_line = 'variance space'


plot_nfs_runs(result_dict, title=f'{first_line}\n{second_line} in {third_line}')
