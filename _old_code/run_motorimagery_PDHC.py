import pickle
import gpflow
import numpy as np
from GPClassificationRunner import GPClassificationRunner
from ch_info_dict import ch_names, ch_location


if __name__ == "__main__":
    # Example arrays — we will split internally
    # X = np.load("./data/motorimagery_X.npy")  # shape (N, s, s)
    # Y = np.load("./data/motorimagery_Y.npy")  # This is a binary class
    # dataset_label = "sub-PDHC002_ses-01_run-01"

    # Load input data: X, Y
    # X = np.load("./data/classification_X.npy")  # shape (N, s, s)
    # Y = np.load("./data/classification_Y.npy")  # This is a binary class
    # dataset_label = "toy_dataset"

    path_to_files = [
        # "sub-PDHC001_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC002_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC002_ses-01_task-MotorImag_run-02.pkl",
        # "sub-PDHC003_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC004_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC005_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC006_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC007_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC008_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC009_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC010_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC011_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC012_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC013_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC014_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC015_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC016_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC017_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC018_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC019_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC020_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC021_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC022_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC023_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC024_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC025_ses-01_task-MotorImag_run-01.pkl",
        # "sub-PDHC026_ses-01_task-MotorImag_run-01.pkl",
        "sub-PDHC027_ses-01_task-MotorImag_run-01.pkl",
    ]

    for path_to_file in path_to_files:
        # Load input data: dict
        with open(f"./data/{path_to_file}", "rb") as f:
            data = pickle.load(f)

        nf = 2

        runner = GPClassificationRunner(
            # Input variables
            X=data["X"],
            Y=data["Y"],
            dataset_label=data["dataset_label"],
            ch_names=data["ch_names"],
            ch_xy=ch_location,
            # Model / kernel
            weights_init="random",
            nf=nf,
            eta_flag=False,
            ard_flag=False,
            logged_flag=True,
            kernel_type="RBF",
            # Training
            frac_train=0.75,
            model_class=gpflow.models.VGP,
            model_kwargs=None,
            likelihood_class=gpflow.likelihoods.Bernoulli,
            likelihood_kwargs=None,
            training_loss_fn=None,
            predict_y_fn=None,
            learning_rate=0.01,
            maxiter=3000,
            pred_threshold=0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
            # random_state=42,
            random_state=np.random.randint(0, 100),
            # GIF controls
            gif_flag=True,  # generate or not gifs
            gif_stride=10,  # sample every k iterations
            gif_max_frames=None,  # auto-raise stride to cap frames
            synced_gif=True,  # build synced dashboard GIF
            topomap_filters_for_gif=nf,  # animate first k filters of W
            # Run naming / Logging
            results_dir="./results",
            run_name=data["dataset_label"],
        )
        runner.run()
