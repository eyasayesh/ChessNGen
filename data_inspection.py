import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import time

class DatasetInspector:
    def __init__(self, h5_files: List[str]):
        self.h5_files = [Path(f) for f in h5_files]

    def inspect_file_metadata(self) -> Dict[str, Any]:
        """Inspect basic metadata of all H5 files"""
        metadata = {}
        total_size = 0
        
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                file_metadata = {
                    'num_images': len(f['images']),
                    'image_shape': f['images'].shape[1:],
                    'dtype': str(f['images'].dtype),
                    'compression': f['images'].compression,
                    'compression_opts': f['images'].compression_opts,
                    'chunks': f['images'].chunks,
                    'file_size_gb': h5_file.stat().st_size / (1024**3)
                }
                metadata[h5_file.name] = file_metadata
                total_size += file_metadata['num_images']
        
        metadata['total_images'] = total_size
        return metadata

    def benchmark_loading_speed(self, batch_size: int = 32, num_batches: int = 10,
                                val_split=0.1,num_workers=3) -> Dict[str, float]:
        """Benchmark data loading speed"""
        from chess_data_module import Chess_Data_Module  # Import your dataset class
        
        data_module = Chess_Data_Module(h5_files=self.h5_files,batch_size=batch_size,val_split=val_split,num_workers=num_workers)
        loader = data_module.train_dataloader()
        
        # Time batch loading
        start_time = time.time()
        times = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            end_time = time.time()
            times.append(end_time - start_time)
            start_time = end_time
            
        return {
            'avg_batch_time': np.mean(times[1:]),  # Skip first batch (warm-up)
            'std_batch_time': np.std(times[1:]),
            'images_per_second': batch_size / np.mean(times[1:])
        }

    def analyze_image_statistics(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Analyze image statistics from random samples"""
        from torch.utils.data import DataLoader
        from your_datamodule import ChessImageDataset
        
        dataset = ChessImageDataset(self.h5_files)
        indices = torch.randperm(len(dataset))[:num_samples]
        
        pixels = []
        for idx in indices:
            img = dataset[idx]
            pixels.append(img.numpy().flatten())
        
        pixels = np.concatenate(pixels)
        
        return {
            'mean': np.mean(pixels),
            'std': np.std(pixels),
            'min': np.min(pixels),
            'max': np.max(pixels),
            'histogram': np.histogram(pixels, bins=50)
        }

    def visualize_random_samples(self, num_samples: int = 16, save_path: Optional[str] = None):
        """Visualize random samples from the dataset"""
        dataset = ChessImageDataset(self.h5_files)
        indices = torch.randperm(len(dataset))[:num_samples]
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for idx, ax in zip(indices, axes.flat):
            img = dataset[idx]
            ax.imshow(img.permute(1, 2, 0))  # Change from CxHxW to HxWxC
            ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def check_data_integrity(self, num_samples: int = 100) -> Dict[str, Any]:
        """Check for potential data issues"""
        dataset = ChessImageDataset(self.h5_files)
        indices = torch.randperm(len(dataset))[:num_samples]
        
        issues = {
            'nan_images': 0,
            'zero_images': 0,
            'wrong_shape': 0,
            'out_of_range': 0
        }
        
        for idx in indices:
            img = dataset[idx]
            
            if torch.isnan(img).any():
                issues['nan_images'] += 1
            
            if (img == 0).all():
                issues['zero_images'] += 1
            
            if img.shape != (3, 256, 256):  # Adjust shape as needed
                issues['wrong_shape'] += 1
            
            if (img < 0).any() or (img > 1).any():
                issues['out_of_range'] += 1
        
        return issues

    def memory_usage_estimate(self) -> Dict[str, float]:
        """Estimate memory usage for different batch sizes"""
        with h5py.File(self.h5_files[0], 'r') as f:
            single_image_size = np.prod(f['images'].shape[1:]) * 4  # 4 bytes for float32
        
        total_images = sum(len(h5py.File(f, 'r')['images']) for f in self.h5_files)
        
        return {
            'single_image_mb': single_image_size / (1024**2),
            'batch_32_mb': (single_image_size * 32) / (1024**2),
            'batch_64_mb': (single_image_size * 64) / (1024**2),
            'full_dataset_gb': (single_image_size * total_images) / (1024**3)
        }

# Example usage
if __name__ == "__main__":
    h5_files = ["chess_data_1.h5", "chess_data_2.h5"]
    inspector = DatasetInspector(h5_files)
    
    # Print file metadata
    print("\nFile Metadata:")
    print(inspector.inspect_file_metadata())
    
    # Benchmark loading speed
    print("\nLoading Speed Benchmark:")
    print(inspector.benchmark_loading_speed())
    
    # Check image statistics
    print("\nImage Statistics:")
    print(inspector.analyze_image_statistics())
    
    # Check for data issues
    print("\nData Integrity Check:")
    print(inspector.check_data_integrity())
    
    # Estimate memory usage
    print("\nMemory Usage Estimates:")
    print(inspector.memory_usage_estimate())
    
    # Visualize samples
    inspector.visualize_random_samples(save_path="sample_images.png")
