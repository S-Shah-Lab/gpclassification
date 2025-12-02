import pickle
import gpflow

# from GPClassificationRunner import GPClassificationRunner
from gp_classification import GPClassificationRunner


spatialFilter_init = "manual"
random_state = 2
nf = 2

path_to_files = [
    "data_set_IVa_aa.pkl",
    # "data_set_IVa_al.pkl",
    # "data_set_IVa_av.pkl",
    # "data_set_IVa_aw.pkl",
    # "data_set_IVa_ay.pkl",
]

for path_to_file in path_to_files:
    # Load input data: dict
    with open(f"./data/{path_to_file}", "rb") as f:
        data = pickle.load(f)

    runner = GPClassificationRunner(
        X=data["X"],
        Y=data["Y"],
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
        maxiter=1400,
        pred_threshold=0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
        random_state=random_state,
        frac_val=0.0,
        frac_test=0.3,
        # ----- Policy flags for adaptation / early stopping
        use_validation_for_adaptation=False,  # if True and val exists, adapt LR/ES on val; else train-only
        enable_adaptation=True,  # enable LR reduce-on-plateau on chosen set
        enable_early_stopping=False,  # enable early stopping on chosen set
        # Run naming / Logging
        results_dir=f"./results/{data["dataset_label"]}",
        run_name=None,
    )
    runner.run()
