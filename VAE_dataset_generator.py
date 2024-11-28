import chess
import chess.pgn
import chess.svg
import cairosvg  # For converting SVG to PNG
import os
import numpy as np
import h5py
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import subprocess



class Chess_Dataset_Generator:
    def __init__(self, output_dir="./vae_chess_images",img_size = 256):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        self.images = []

        # Define custom CSS style for squares
        self.css_style = """
            .square {
                stroke: #333333;
                stroke-width: 1px;
            }
        """

    
    #helper method for getting number of lines
    def get_total_games(pgn_path):
        # Use wc -l to count lines efficiently
        result = subprocess.run(['wc', '-l', pgn_path], capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0])
        return total_lines // 40
        
    def generate_dataset_from_pgn(self, pgn_path, dataset_name, positions_to_capture="all", 
                                  max_positions_per_game=None, max_num_games = None, max_games_per_chunk = 1000):
        """
        Process a PGN file and generate images for specified positions
        
        Args:
            pgn_path (str): Path to PGN file
            positions_to_capture (str or list): 
                - "all" for every position
                - list of move numbers
                - "first_last" for first and last position
            max_positions_per_game (int): Maximum positions to capture per game
            max_number_games: maximum number of games to convert to images
            max_games_per_chunk: max number of games in one h5 file
        """
       
        with open(pgn_path) as pgn_file:
            game_count = 0
            chunk_count = 1
            # Set max_num_games based on file size if None
            if max_num_games is None:
                max_num_games = self.get_total_games(pgn_path)

            with tqdm(total=max_num_games, desc="Processing games",leave=False) as outer_bar:
                while True:
                    for chunck in tqdm(range(max_games_per_chunk), 
                                        total = max_games_per_chunk,
                                        desc = f"Chunk #{chunk_count}",
                                        leave = False):
                            game = chess.pgn.read_game(pgn_file)
                            if game is None or game_count == max_num_games:  # End of file
                                break
                            game_count += 1
                            game_id = f"game_{game_count}"
                        
                            self.process_single_game(game, game_id, positions_to_capture, max_positions_per_game)
                            outer_bar.update(1)
                    if game is None or game_count == max_num_games:  # End of file
                        self.save_dataset(f'{self.output_dir}/{dataset_name}_final_chunk.h5')
                        break
                    self.save_dataset(f'{self.output_dir}/{dataset_name}_chunk_{chunk_count}.h5')
                    chunk_count += 1
                
    def process_single_game(self, game, game_id, positions_to_capture, max_positions_per_game):
        """Process a single game and generate images"""
        board = game.board()
        moves = list(game.mainline_moves())
        total_moves = len(moves)
        
        # Determine which positions to capture
        positions = self._get_positions_to_capture(
            total_moves, positions_to_capture, max_positions_per_game
        )
        
        # Generate initial position
        if 0 in positions:
            self._board_to_array(board)
        
        # Play through the game
        for move_num, move in tqdm(enumerate(moves, 1), 
                          total=len(moves), 
                          desc=f"Processing moves in game {game_id}", 
                          leave=False):  # leave=False makes the bar disappear after finishing
            board.push(move)
            if move_num in positions:
                img_array = self._board_to_array(board)
                self.images.append(img_array)

    def _get_positions_to_capture(self, total_moves, positions_to_capture, max_positions):
        """Determine which positions should be captured"""
        if positions_to_capture == "all":
            positions = set(range(total_moves + 1))
        elif positions_to_capture == "first_last":
            positions = {0, total_moves}
        elif isinstance(positions_to_capture, list):
            positions = set(pos for pos in positions_to_capture if pos <= total_moves)
        else:
            raise ValueError("Invalid positions_to_capture specification")
            
        if max_positions and len(positions) > max_positions:
            positions = set(sorted(positions)[:max_positions])
            
        return positions
        
    def _board_to_array(self, board):
        """Generate and save board image"""

        # Create SVG
        svg_data = chess.svg.board(
        board,
        size=self.img_size,
        style=self.css_style,     
        coordinates = False,
        borders = True
    )
            
        # Convert SVG to PNG in memory
        board_move_png =  cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

        # Convert PNG data to numpy array
        move_img = Image.open(BytesIO(board_move_png))
        move_img = move_img.convert('RGB')  # Convert from RGBA to RGB
        img_array = np.array(move_img)
        
        # Convert to PyTorch format (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
    
    def save_dataset(self,path_to_dataset):
        """Save all processed images to H5 file"""
        print("Converting to numpy array...")
        image_data = np.array(self.images, dtype=np.uint8)
        
        print(f"Saving dataset of shape {image_data.shape} to {path_to_dataset}")
        with h5py.File(path_to_dataset, 'w') as f:
            f.create_dataset('images', data=image_data, compression='gzip')
        
        # Clear memory
        self.images = []
        print("Dataset saved successfully!")

# Example usage
if __name__ == "__main__":
    generator = Chess_Dataset_Generator(output_dir='~/scratch/vae_chess_data')

    generator.generate_dataset_from_pgn("./game_pgns/ficsgamesdb_2022_standard2000_nomovetimes_403053.pgn",'test',
                                        max_num_games =14,max_games_per_chunk=3)
    
    