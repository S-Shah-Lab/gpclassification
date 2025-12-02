import pickle
from sklearn.model_selection import train_test_split

from gp_classification_gpy2 import GPClassificationRunner

# File to run on
path_to_file = "data_set_IVa_aa.pkl"
# path_to_file = "data_set_IVa_al.pkl"
# path_to_file = "data_set_IVa_av.pkl"
# path_to_file = "data_set_IVa_aw.pkl"
# path_to_file = "data_set_IVa_ay.pkl"

# Load input data: dict
with open(f"../data/{path_to_file}", "rb") as f:
    data = pickle.load(f)

# Consider train fraction from entire dataset, no validation, the remaining is test
train_splits = [0.7]  # [0.7, 0.9, 0.5, 0.3, 0.1]

# Consider random states for random-influenced events
# - Initialization of W matrix if `spatialFilter_init` is set to `random`
spatialFilter_init = "ones"
random_states = [11]  # [0, 11, 22, 33, 44, 55]

# Number of spatial filters to consider
nfs = [2]  # [1, 2, 4, 8, 12, 16, 20, 30]

for train_split in train_splits:
    # Fix train and test split to be trained over many times
    # This train / test split is fixed, given the same train fraction
    # the data in train will always be the same
    X_train, X_test, Y_train, Y_test = train_test_split(
        data["X"], data["Y"], train_size=train_split, random_state=2
    )

    # Generate dictionary for the GP runner (explicit train/test splits)
    my_dict = {
        "X": {
            "train": X_train,
            "test": X_test,
        },
        "Y": {
            "train": Y_train,
            "test": Y_test,
        },
    }

    for nf in nfs:
        for random_state in random_states:
            results_dir = (
                f"/mnt/c/Users/scana/Desktop/gpc/gpy_results/"
                f"{data['dataset_label']}/"
                f"seed_{spatialFilter_init}/"
                f"split_{train_split}/"
                f"nf_{nf}"
            )
            
            runner = GPClassificationRunner(
                X=my_dict["X"],
                Y=my_dict["Y"],
                dataset_label=data["dataset_label"],
                ch_names=data["ch_names"],
                ch_xy=data["ch_location"],
                spatialFilter_init=spatialFilter_init,
                nf=nf,
                eta_flag=False,
                ard_flag=False,
                W_trainable=True,
                logged_flag=True,
                kernel_type="RBF",              # 'Linear', 'RBF'
                maxiter=300,                    # Max allowed EP optimization steps
                frac_val=0.0,
                frac_test=0.0,
                random_state=random_state,
                results_dir=results_dir,
                run_name=None,
            )
            runner.fit()
            
            # probabilities on each split
            #p_tr = runner.predict_proba("train")
            #p_va = runner.predict_proba("val")
            #p_te = runner.predict_proba("test")

            # JSON-able training log if you want to save/inspect
            #log = runner.to_json()
            #print(f"Finished. Steps: {len(log['logs'])}, best step: {runner.best_step}")
