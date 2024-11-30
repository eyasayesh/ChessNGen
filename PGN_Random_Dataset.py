import chess
import chess.pgn
import chess.svg
import cairosvg
import os
import numpy as np
import h5py
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import subprocess
import math

class PGN_Random_Dataset:
    def __init__(self, output_dir="./shuffled_chess_datasets", img_size=256):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        self.buffer = []
        self.current_dataset = []
        self.dataset_counter = 0

        self.css_style = """
            .square {
                stroke: #333333;
                stroke-width: 1px;
            }
        """

    def estimate_total_moves(self, pgn_path):
        total_moves = 0
        with open(pgn_path) as pgn_file:
            # Sample first 100 games to estimate average moves per game
            games_sampled = 0
            moves_sampled = 0
            
            for _ in range(100):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                moves_sampled += len(list(game.mainline_moves()))
                games_sampled += 1
            
            if games_sampled == 0:
                return 0
                
            avg_moves_per_game = moves_sampled / games_sampled
            
            # Get total number of games
            total_games = self.estimate_total_games(pgn_path)
            
            # Estimate total moves
            total_moves = int(avg_moves_per_game * total_games)
            
        return total_moves

    def estimate_total_games(self, pgn_path):
        result = subprocess.run(['wc', '-l', pgn_path], capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0])
        return total_lines // 40

    def sample_from_buffer(self, num_samples):
        """Sample positions from buffer without replacement"""
        if len(self.buffer) < num_samples:
            samples = self.buffer.copy()
            np.random.shuffle(samples)
            self.buffer = []
            return samples
        
        # Convert buffer to numpy array for efficient indexing
        buffer_array = np.array(self.buffer)
        
        # Get random indices
        indices = np.random.choice(len(buffer_array), num_samples, replace=False)
        
        # Create boolean mask for remaining elements
        mask = np.ones(len(buffer_array), dtype=bool)
        mask[indices] = False
        
        # Extract samples and update buffer efficiently
        samples = buffer_array[indices].tolist()
        self.buffer = buffer_array[mask].tolist()
        
        return samples

    def generate_dataset_from_pgn(self, pgn_path, dataset_name, max_num_datasets=None, 
                                max_moves_per_dataset=10000, buffer_size_ratio=0.1):
        """
        Process a PGN file and generate multiple datasets with buffered sampling
        
        Args:
            pgn_path (str): Path to PGN file
            dataset_name (str): Base name for output datasets
            max_num_datasets (int): Maximum number of datasets to generate
            max_moves_per_dataset (int): Maximum moves per dataset
            buffer_size_ratio (float): Size of buffer relative to max_moves_per_dataset
        """
        # Calculate total moves and adjust max_num_datasets if needed
        total_moves = self.estimate_total_moves(pgn_path)
        possible_datasets = total_moves // max_moves_per_dataset
        
        if max_num_datasets is None:
            max_num_datasets = possible_datasets
        else:
            max_num_datasets = min(max_num_datasets, possible_datasets)

        buffer_size = int(max_moves_per_dataset * buffer_size_ratio)
        
        print(f"Creating {max_num_datasets} datasets with {max_moves_per_dataset} moves each")
        print(f"Buffer size: {buffer_size}")
        
        with open(pgn_path) as pgn_file:
            with tqdm(total=max_num_datasets * max_moves_per_dataset, 
                     desc="Total Progress") as outer_bar:
                
                while self.dataset_counter < max_num_datasets:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        # End of file reached
                        break
                    
                    # Process the game
                    self.process_single_game(game, buffer_size)
                    
                    # Check if buffer is full enough to sample
                    while len(self.buffer) >= buffer_size:
                        # Calculate how many more positions we need for current dataset
                        remaining_needed = max_moves_per_dataset - len(self.current_dataset)
                        sample_size = min(buffer_size // 2, remaining_needed)
                        
                        if sample_size > 0:
                            samples = self.sample_from_buffer(sample_size)
                            self.current_dataset.extend(samples)
                            outer_bar.update(len(samples))
                        
                        # Check if dataset is complete
                        if len(self.current_dataset) >= max_moves_per_dataset:
                            self.save_dataset(
                                f'{self.output_dir}/{dataset_name}_dataset_{self.dataset_counter}.h5'
                            )
                            self.current_dataset = []
                            self.dataset_counter += 1
                            
                            if self.dataset_counter >= max_num_datasets:
                                break
                
                # Save any remaining data if we have enough for a partial dataset
                if len(self.current_dataset) > max_moves_per_dataset // 2:
                    self.save_dataset(
                        f'{self.output_dir}/{dataset_name}_dataset_final.h5'
                    )

    def process_single_game(self, game, buffer_size):
        """Process a single game and add positions to buffer"""
        board = game.board()
        
        # Add initial position
        self.buffer.append(self._board_to_array(board))
        if len(self.buffer) > buffer_size * 2:  # Allow some overflow for better mixing
            return
            
        # Process moves
        for move in game.mainline_moves():
            board.push(move)
            self.buffer.append(self._board_to_array(board))
            if len(self.buffer) > buffer_size * 2:
                return

    def _board_to_array(self, board):
        """Generate board image as numpy array"""
        svg_data = chess.svg.board(
            board,
            size=self.img_size,
            style=self.css_style,     
            coordinates=False,
            borders=True
        )
        
        board_move_png = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        move_img = Image.open(BytesIO(board_move_png))
        move_img = move_img.convert('RGB')
        img_array = np.array(move_img)
        
        # Convert to PyTorch format (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
    
    def save_dataset(self, path_to_dataset):
        """Save current dataset to H5 file"""
        print(f"\nSaving dataset {self.dataset_counter} with {len(self.current_dataset)} positions")
        image_data = np.array(self.current_dataset, dtype=np.uint8)
        
        with h5py.File(path_to_dataset, 'w') as f:
            dset = f.create_dataset('images', data=image_data, compression='gzip')

            # Add dataset-level metadata
            dset.attrs['num_positions'] = len(image_data)
            dset.attrs['image_size'] = self.img_size
            dset.attrs['channels'] = image_data.shape[1]
            dset.attrs['height'] = image_data.shape[2]
            dset.attrs['width'] = image_data.shape[3]
            dset.attrs['format'] = 'CHW'  # Channel, Height, Width format
            dset.attrs['size in MB'] = np.prod(image_data.shape)//1e6
            
          
            # Create a metadata group for additional information
            meta = f.create_group('metadata')
            meta.attrs['css_style'] = np.string_(self.css_style)
            meta.attrs['coordinates_enabled'] = False
            meta.attrs['borders_enabled'] = True
                   
        print(f"Dataset saved to {path_to_dataset} with full metadata")

    @staticmethod
    def read_metadata(h5_path):
        """Read metadata from an H5 file"""
        metadata = {}
        with h5py.File(h5_path, 'r') as f:
            # Read dataset attributes
            metadata['dataset_attrs'] = dict(f['images'].attrs)
            
            # Read metadata group
            if 'metadata' in f:
                metadata['metadata'] = {
                    'general': dict(f['metadata'].attrs)
                }
            
            # Add basic dataset information
            metadata['shape'] = f['images'].shape
            metadata['dtype'] = str(f['images'].dtype)
            metadata['chunks'] = f['images'].chunks
            metadata['compression'] = f['images'].compression
            
        return metadata

# Example usage
if __name__ == "__main__":
    generator = PGN_Random_Dataset(output_dir='./chess_datasets')
    generator.generate_dataset_from_pgn(
        "./game_pgns/ficsgamesdb_202202_standard_nomovetimes_403064.pgn",
        'test_positions',
        max_num_datasets=3,
        max_moves_per_dataset=1000,
        buffer_size_ratio=0.1
    )