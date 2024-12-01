import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from PGN_Random_Dataset import PGN_Random_Dataset

def verify_images(h5_file, num_samples=5):
    with h5py.File(h5_file, 'r') as f:
        images = f['images']
        total_images = len(images)
        
        # Randomly sample some images
        indices = np.random.choice(total_images, num_samples)
        
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i+1)
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(images[idx], (1, 2, 0))
            plt.imshow(img)
            plt.title(f"Image {idx}")
        plt.show()

def save_sample_images(h5_file, num_samples=5, output_dir='sample_images'):
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as f:
        images = f['images']
        indices = np.random.choice(len(images), num_samples)
        
        for i, idx in enumerate(indices):
            img = images[idx]
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            
            plt.imsave(f'{output_dir}/sample_{idx}.png', img)

if __name__ == "__main__":
   #verify_images('./vae_chess_images/test.h5', num_samples=5)
    save_sample_images('/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_7.h5', num_samples=5)
    print(PGN_Random_Dataset.read_metadata('/home/hice1/eayesh3/scratch/vae_datasets/rand_chess_pos_dataset_dataset_7.h5'))