import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from pathlib import Path
from typing import List, Optional

class Chess_Image_Dataset(Dataset):
    def __init__(self, h5_files: List[str]):
        """
        Creats a Dataset from chess images stored in H5 files
        
        Args:
            h5_files: List of paths to H5 files
        """
        self.h5_files = [Path(f) for f in h5_files]
        # Keep file handles open during dataset lifetime
        self.file_handles = []
        
        # Get total size and file mappings
        self.size = 0
        self.file_mappings = []  # (file_idx, internal_idx)
        
        for file_idx, h5_file in enumerate(self.h5_files):
            handle = h5py.File(h5_file, 'r')
            self.file_handles.append(handle)
            n_images = len(handle['images'])
            self.file_mappings.extend([(file_idx, i) for i in range(n_images)])
            self.size += n_images
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, internal_idx = self.file_mappings[idx]
        
        # Get image from open file handle instead of opening/closing
        image = self.file_handles[file_idx]['images'][internal_idx]
        
        # Convert to torch tensor and normalize
        image = torch.from_numpy(image).float()
        image = image / 255.0  # Normalize to [0, 1]
            
        return image

    
    # CHANGE: Added cleanup method
    def __del__(self):
        """Cleanup open file handles"""
        for handle in self.file_handles:
            handle.close()

class Chess_Data_Module:
    def __init__(
        self,
        h5_files: List[str],
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42
    ):
        """
        DataModule for chess image dataset
        
        Args:
            h5_files: List of H5 file paths
            batch_size: Batch size for dataloaders
            val_split: Fraction of data to use for validation
            num_workers: Number of workers for dataloaders
            seed: Random seed for reproducibility
        """
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.setup()
    
    def setup(self):
        """Prepare train and validation datasets"""
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Create full dataset
        full_dataset = Chess_Image_Dataset(self.h5_files)
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Generate random indices for splits
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train and validation datasets using Subset
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        
        print(f"Dataset split complete:")
        print(f"Total images: {dataset_size}")
        print(f"Training images: {train_size}")
        print(f"Validation images: {val_size}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Example usage:
if __name__ == "__main__":
    # Create data module with just one or two files
    h5_files = ["/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_0.h5", 
                "/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_4.h5"]
    data_module = Chess_Data_Module(
        h5_files=h5_files,
        batch_size=32,
        val_split=0.1,
        num_workers=3
    )
    
    # Setup and get loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
