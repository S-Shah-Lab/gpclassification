import pickle
import gpflow

# from GPClassificationRunner import GPClassificationRunner
from gp_classification import GPClassificationRunner
from sklearn.model_selection import train_test_split

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
train_splits = [0.7] # [0.7, 0.9, 0.5, 0.3, 0.1]

# Consider random states for random-influenced events
# - Initialization of W matrix if `spatialFilter_init` is set to `random`
# - train/test split if arrays are passed instead of dictionaries
spatialFilter_init = "random"
random_states = [0] # [0, 11, 22, 33, 44, 55]

# Number of spatial filters to consider
nfs = [1, 2]  # [2, 4, 8, 12, 16, 20, 30]

for train_split in train_splits:
    # Fix train and test split to be trained over many times
    # This train / test split is fixed, given the same train fraction the data in train will always be the same
    X_train, X_test, Y_train, Y_test = train_test_split(
        data["X"], data["Y"], train_size=train_split, random_state=2
    )

    # Generate dictionary
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
            runner = GPClassificationRunner(
                # Input variables
                X=my_dict["X"],
                Y=my_dict["Y"],
                dataset_label=data["dataset_label"],
                ch_names=data["ch_names"],
                ch_xy=data["ch_location"],
                # Model / kernel
                spatialFilter_init=spatialFilter_init,
                nf=nf,
                eta_flag=False,
                ard_flag=False,
                logged_flag=True,
                kernel_type="RBF",
                model_class=gpflow.models.VGP,
                likelihood_class=gpflow.likelihoods.Bernoulli,
                # Training
                learning_rate=0.1,  # Adam default learning rate
                gamma=0.1,  # Natural gradient default learning rate
                maxiter=1800,
                pred_threshold=0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
                random_state=random_state,
                # ----- Policy flags for adaptation / early stopping
                use_validation_for_adaptation=False,  # if True and val exists, adapt LR/ES on val; else train-only
                enable_adaptation=True,  # enable LR reduce-on-plateau on chosen set
                enable_early_stopping=False,  # enable early stopping on chosen set
                # Run naming / Logging
                results_dir=f"/mnt/c/Users/scana/Desktop/gpc/results/{data["dataset_label"]}/seed_{spatialFilter_init}/split_{train_split}/nf_{nf}",
                run_name=None,
            )
            runner.run()
