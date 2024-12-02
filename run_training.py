from train import train_vae
from vae_basic import VAEConfig

# Create configuration with specified hyperparameters
config = VAEConfig(
    input_size=256,
    in_channels=3,
    latent_dim=512,
    hidden_dims=[32, 64, 128, 256],
    max_lr=0.0005,
    kld_weight=0.0000001,
    negative_slope=0.1,
    weight_decay=0.000001,
    epochs=15,
    minibatch_size=64,
    minibatch_num=4,
    val_split=0.1,
    num_workers=3,
    save_epochs=10,
    warmup_ratio=0.1,
    div_factor=25,
    final_div_factor=1000,
    anneal_strategy='cos'
)

# Run training with specified H5 files
h5_files = [
    "/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_0.h5"]

train_vae(
    config=config,
    h5_files=h5_files,
    run_name="tangerine_4_512",
    checkpoint_dir="/home/hice1/eayesh3/scratch"
)

