#!/bin/bash

#SBATCH --job-name=tangerine_5_512
#SBATCH --output=./logs/tangerine_5_512_%A.log    # %A = job ID, %a = array task ID
#SBATCH --error=./errors/tangerine_5_512_%A.err
#SBATCH --time=05:00:00                  # 3 hour time limit
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=4                # Request 4 CPUs
#SBATCH --mem=32G                        # Request 16GB RAM


# Print some debugging information
echo "Starting job array ${SLURM_ARRAY_TASK_ID}"
echo "Running on host $(hostname)"
echo "Starting at $(date)"

# Load necessary modules (adjust these based on your system)
module purge  # Clear any loaded modules
module load anaconda3
conda activate cAI8803-gpu

# Define paths to H5 files
H5_FILES=(
    "/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_0.h5"
)

# Define checkpoint directory
CHECKPOINT_DIR="/home/hice1/eayesh3/scratch"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Configure hyperparameters
RUN_NAME="tangerine_5_512"
INPUT_SIZE=256
IN_CHANNELS=3
LATENT_DIM=512
HIDDEN_DIMS=(32 64 128 256)  # Space-separated list of dimensions
MAX_LR=0.0005
KLD_WEIGHT=0.00000005
NEGATIVE_SLOPE=0.1
WEIGHT_DECAY=0.000001
EPOCHS=15
MINIBATCH_SIZE=64
MINIBATCH_NUM=4
VAL_SPLIT=0.1
NUM_WORKERS=3
SAVE_EPOCHS=10
WARMUP_RATIO=0.1
DIV_FACTOR=25
FINAL_DIV_FACTOR=1000
ANNEAL_STRATEGY="cos"



# Convert HIDDEN_DIMS array to Python list string
HIDDEN_DIMS_STR="[$(echo ${HIDDEN_DIMS[@]} | sed 's/ /, /g')]"

# Create Python script for configuration and training
cat << EOF > run_training.py
from train import train_vae
from vae_basic import VAEConfig

# Create configuration with specified hyperparameters
config = VAEConfig(
    input_size=$INPUT_SIZE,
    in_channels=$IN_CHANNELS,
    latent_dim=$LATENT_DIM,
    hidden_dims=$HIDDEN_DIMS_STR,
    max_lr=$MAX_LR,
    kld_weight=$KLD_WEIGHT,
    negative_slope=$NEGATIVE_SLOPE,
    weight_decay=$WEIGHT_DECAY,
    epochs=$EPOCHS,
    minibatch_size=$MINIBATCH_SIZE,
    minibatch_num=$MINIBATCH_NUM,
    val_split=$VAL_SPLIT,
    num_workers=$NUM_WORKERS,
    save_epochs=$SAVE_EPOCHS,
    warmup_ratio=$WARMUP_RATIO,
    div_factor=$DIV_FACTOR,
    final_div_factor=$FINAL_DIV_FACTOR,
    anneal_strategy='$ANNEAL_STRATEGY'
)

# Run training with specified H5 files
h5_files = [
    $(printf '"%s",' "${H5_FILES[@]}" | sed 's/,$//')]

train_vae(
    config=config,
    h5_files=h5_files,
    run_name="$RUN_NAME",
    checkpoint_dir="$CHECKPOINT_DIR"
)
EOF

# Run the training
echo "Starting VAE training..."
python run_training.py

# Clean up temporary Python script
rm run_training.py

echo "Training complete!"