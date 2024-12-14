import torch
import os, sys


class TimeGANSettings:
    def __init__(self):
        # Data and model architecture
        self.data_name = ""  # data path for windowed data, npy array of activitys df
        self.activity = 0  # index of the activity
        self.seconds = 6
        self.sampling_rate = 100
        self.seq_len = self.sampling_rate * self.seconds
        self.columns = ["x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro"]
        self.z_dim = len(self.columns)
        self.module = "gru"  # choices: 'gru', 'lstm', 'lstmLN'
        self.hidden_dim = 24
        self.num_layer = 3

        # Training parameters
        self.iteration = 1  # 50000
        self.batch_size = 128
        self.metric_iteration = 10

        # Computation resources
        self.workers = 8
        self.device = "gpu"
        self.gpu_ids = "0"
        self.ngpu = 1

        # Model and output
        self.model = "TimeGAN"
        self.outf = ""
        self.name = "experiment_name"

        # Display options
        self.display_server = "http://localhost"
        self.display_port = 8097
        self.display_id = 0
        self.display = False

        # Misc
        self.manualseed = -1

        # Training specifics
        self.print_freq = 100
        self.load_weights = False
        self.resume = ""
        self.beta1 = 0.9
        self.lr = 0.001

        # Loss weights
        self.w_gamma = 1
        self.w_es = 0.1
        self.w_e0 = 10
        self.w_g = 100

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def setup_device(self):
        """Setup device based on availability. Prefer GPU if available."""
        if self.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use first available GPU
            print("Using GPU for computation.")
        else:
            self.device = torch.device("cpu")  # Fall back to CPU
            print("GPU not available. Using CPU for computation.")
        return self.device

    def create_output_dir(self):
        """Create output directory for saving experiment results."""
        if self.name == "experiment_name":
            self.name = f"{self.model}/{self.data_name}"  # Modify name if default
        expr_dir = os.path.join(self.outf, self.name)
        try:
            os.makedirs(expr_dir, exist_ok=True)  # Create directory if not exists
        except OSError as e:
            print(f"Error creating directory {expr_dir}: {e}")
            sys.exit(1)
        return expr_dir

    def save_options(self):
        """Save the options to a file."""
        expr_dir = self.create_output_dir()
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(vars(self).items()):
                opt_file.write(f"{str(k)}: {str(v)}\n")
            opt_file.write("-------------- End ----------------\n")

    def set_manual_seed(self):
        """Set manual seed for reproducibility."""
        if self.manualseed != -1:
            torch.manual_seed(self.manualseed)
            torch.cuda.manual_seed_all(self.manualseed)
            print(f"Manual seed set to {self.manualseed}")
        else:
            print("No manual seed provided.")

    def is_train(self, train_mode=True):
        """Set training or testing mode."""
        self.isTrain = train_mode
        print(f"Mode set to {'training' if self.isTrain else 'testing'} mode.")
