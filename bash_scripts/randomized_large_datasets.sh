#!/bin/bash

#SBATCH --job-name=rand_large_dataset
#SBATCH --output=./logs/rand_large_dataset_%A.log    # %A = job ID, %a = array task ID
#SBATCH --error=./errors/rand_large_dataset_%A.err
#SBATCH --time=07:30:00                  # 6 hour time limit
#SBATCH --cpus-per-task=1                # Request 1 CPUs
#SBATCH --mem=24G                        # Request 24GB RAM


# Print some debugging information
echo "Starting job array ${SLURM_ARRAY_TASK_ID}"
echo "Running on host $(hostname)"
echo "Starting at $(date)"

# Load necessary modules (adjust these based on your system)
module purge  # Clear any loaded modules
module load anaconda3
conda activate cAI8803-gpu

# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set variables
PGN_FILE="./game_pgns/ficsgamesdb_2022_standard2000_nomovetimes_403053.pgn"
OUTPUT_DIR="/home/hice1/eayesh3/scratch/vae_datasets"
DATASET_NAME="rand_chess_pos_dataset"
MAX_NUM_DATASETS=20  # 200,000 images
MAX_MOVES_PER_DATASET=10000
BUFFER_SIZE_RATIO=0.1

echo "Processing Datasets"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the Python script

python <<END
from PGN_Random_Dataset import PGN_Random_Dataset
import sys
import os

try:
    print("does this do anythin")
    generator = PGN_Random_Dataset(output_dir="$OUTPUT_DIR")
    generator.generate_dataset_from_pgn(
        "$PGN_FILE",
        "$DATASET_NAME",
        max_num_datasets=$MAX_NUM_DATASETS,
        max_moves_per_dataset=$MAX_MOVES_PER_DATASET,
        buffer_size_ratio=$BUFFER_SIZE_RATIO
    )

    print("Successfully completed processing")
except Exception as e:
    print(f"Error in task: {str(e)}", file=sys.stderr)
    sys.exit(1)
END

echo "Job finished at $(date)"
