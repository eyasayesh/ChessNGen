import torch
import torch.nn.functional as F
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools
from vae_basic import ChessVAE, VAEConfig
from chess_data_module import Chess_Data_Module

def log_reconstructions(model, val_batch, epoch):
    """Log image reconstructions to W&B"""
    model.eval()
    with torch.no_grad():
        # Get reconstructions
        recon_images, _, _ = model(val_batch)
        
        # Convert to numpy and scale to [0, 255]
        original = val_batch.cpu().numpy() * 255
        reconstructed = recon_images.cpu().numpy() * 255
        
        # Log a few examples
        images = []
        for i in range(min(4, len(original))):
            # Original image
            orig_img = wandb.Image(
                original[i].transpose(1, 2, 0).astype(np.uint8),
                caption=f"Original {i+1}"
            )
            # Reconstructed image
            recon_img = wandb.Image(
                reconstructed[i].transpose(1, 2, 0).astype(np.uint8),
                caption=f"Reconstructed {i+1}"
            )
            images.extend([orig_img, recon_img])
        
        wandb.log({
            "reconstructions": images,
            "epoch": epoch
        })

def train_vae(config: VAEConfig, h5_files, run_name=None, checkpoint_dir="checkpoints", resume_from=None):
    """
    Train the VAE model with W&B logging and checkpointing
    
    Args:
        config: VAEConfig object containing model and training parameters
        h5_files: List of H5 files containing the chess images
        run_name: Optional name for the W&B run
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume training from
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Initialize W&B run
    wandb.init(
        project="chess-vae",
        name=run_name,
        config={
            "architecture": "VAE",
            "input_size": config.input_size,
            "latent_dim": config.latent_dim,
            "hidden_dims": config.hidden_dims,
            "max_lr": config.max_lr,
            "kld_weight": config.kld_weight,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "negative_slope": config.negative_slope,
            "adamw_betas": config.adamw_betas,
            "minibatch_size": config.minibatch_size,
            "batch_size": config.minibatch_size*config.minibatch_num
        }
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and move to device
    model = ChessVAE(config).to(device)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    # Set up data module
    data_module = Chess_Data_Module(
        h5_files=h5_files,
        batch_size=config.minibatch_size,
        val_split=config.val_split,
        num_workers=config.num_workers
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Configure optimizer and scheduler
    optimizer_config = model.configure_optimizers(train_loader)
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        

        # Training step with gradient accumulation
        accumulation_steps = config.minibatch_num  # Effective batch size will be batch_size * accumulation_steps
        optimizer.zero_grad()

        # Training step
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(device)
                
                # Forward pass
                recon_batch, mu, log_var = model(batch)
                
                # Calculate losses
                recon_loss = F.mse_loss(recon_batch, batch, reduction='mean')
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
                loss = recon_loss + config.kld_weight * kld_loss

                # Normalize loss to account for accumulation
                loss = loss / accumulation_steps  
                
                # Backward pass
                loss.backward()

                
                # Update metrics
                train_loss += loss.item() * accumulation_steps
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()
                
                # Step optimization after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()


                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() * accumulation_steps,
                    'recon_loss': recon_loss.item(),
                    'kld_loss': kld_loss.item()
                })
        
        # Calculate average training losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_kld_loss = train_kld_loss / len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_batch, mu, log_var = model(batch)
                
                # Calculate losses
                recon_loss = F.mse_loss(recon_batch, batch, reduction='mean')
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
                loss = recon_loss + config.kld_weight * kld_loss
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kld_loss = val_kld_loss / len(val_loader)
        
        # Log metrics to W&B
        wandb.log({
            "train/loss": avg_train_loss,
            "train/recon_loss": avg_train_recon_loss,
            "train/kld_loss": avg_train_kld_loss,
            "val/loss": avg_val_loss,
            "val/recon_loss": avg_val_recon_loss,
            "val/kld_loss": avg_val_kld_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch
        })
        
        # Log reconstructions every 5 epochs
        if (epoch + 1) % config.save_epochs == 0:
            log_reconstructions(model, next(iter(val_loader)).to(device), epoch)
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % config.save_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoints/{wandb.run.name}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            
            # Log checkpoint to W&B
            wandb.save(str(checkpoint_path),base_path=checkpoint_dir)
            
            # Keep only the last 3 checkpoints to save space
            checkpoint_files = sorted(checkpoint_dir.glob(f"checkpoints/{wandb.run.name}/checkpoint_epoch_*.pt"))
            if len(checkpoint_files) > 3:
                checkpoint_files[0].unlink()  # Remove oldest checkpoint
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / f"checkpoints/{wandb.run.name}/best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, best_model_path)
            
            # Log best model to W&B
            wandb.save(str(best_model_path),base_path=checkpoint_dir)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, KLD: {avg_train_kld_loss:.4f})")
        print(f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KLD: {avg_val_kld_loss:.4f})")
    
    wandb.finish()
    return best_val_loss

def hyperparameter_sweep():
    """Run hyperparameter sweep with different configurations"""
    # Define hyperparameter search space
    hp_space = {
        'latent_dim': [128, 256, 512],
        'hidden_dims': [
            [32, 64, 128, 256],
            [64, 128, 256, 512],
            [32, 64, 128, 256, 512]
        ],
        'max_lr': [1e-4, 1e-3, 5e-3],
        'kld_weight': [0.1, 0.2, 0.5],
        'negative_slope': [0.01, 0.1],
        'weight_decay': [1e-6, 1e-5]
    }
    
    # Path to your H5 files
    h5_files = [
        "/path/to/your/chess/dataset_0.h5",
        "/path/to/your/chess/dataset_1.h5"
    ]
    
    # Create all combinations of hyperparameters
    keys, values = zip(*hp_space.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Run experiments
    results = []
    for i, hp in enumerate(hp_combinations):
        print(f"\nRunning experiment {i+1}/{len(hp_combinations)}")
        print("Hyperparameters:", hp)
        
        # Create config with current hyperparameters
        config = VAEConfig(
            input_size=256,  # Fixed parameters
            in_channels=3,
            **hp  # Variable hyperparameters
        )
        
        # Create meaningful run name
        run_name = f"vae_lat{hp['latent_dim']}_lr{hp['max_lr']}_kld{hp['kld_weight']}"
        
        # Train model with current configuration
        try:
            best_val_loss = train_vae(config, h5_files, run_name=run_name)
            results.append({
                'hp': hp,
                'best_val_loss': best_val_loss,
                'status': 'completed'
            })
        except Exception as e:
            print(f"Experiment failed with error: {str(e)}")
            results.append({
                'hp': hp,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary of results
    print("\nHyperparameter Sweep Summary:")
    completed_runs = [r for r in results if r['status'] == 'completed']
    if completed_runs:
        best_run = min(completed_runs, key=lambda x: x['best_val_loss'])
        print("\nBest hyperparameters:")
        for k, v in best_run['hp'].items():
            print(f"{k}: {v}")
        print(f"Best validation loss: {best_run['best_val_loss']:.4f}")
    
    # Print failed runs if any
    failed_runs = [r for r in results if r['status'] == 'failed']
    if failed_runs:
        print(f"\nNumber of failed runs: {len(failed_runs)}")
        for r in failed_runs:
            print(f"Failed configuration: {r['hp']}")
            print(f"Error: {r['error']}")

#for testing and demo purposes
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--h5_files', nargs='+', help='Path to H5 files', required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str,
                      help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    if args.sweep:
        hyperparameter_sweep()
    else:
        # Run single training with default configuration
        config = VAEConfig(
            input_size=256,
            latent_dim=256,
            hidden_dims=[32, 64, 128, 256],
            max_lr=1e-3,
            kld_weight=0.2,
            epochs=3,
            save_epochs=1
        )
        train_vae(config, args.h5_files, 
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from)