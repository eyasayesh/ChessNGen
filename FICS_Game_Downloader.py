import requests
from bs4 import BeautifulSoup
import bz2
import os

class FICS_Game_Downloader:
    def __init__(self):
        #https://www.ficsgames.org/download.html -> for formatting
        self.base_url = "https://www.ficsgames.org/cgi-bin/download.cgi"
        
    def download_games(self, game_type = 4, player = '', year = 2022, month = 0, move_times = 0, 
                       download = "Download", game_dir = "./game_pgns"):
        """
        **gametypes**
        1 -> Chess (all time controls, all ratings)
        2 -> Chess (all time controls, average rating > 2000)
        3 -> Standard (all ratings)
        4 -> Standard (average rating > 2000)
        5 -> Blitz (all ratings)
        6 -> Blitz (average rating > 2000)
        7 -> Lightning (all ratings)
        8 -> Lightning (average rating > 2000)
        9 -> Variant games
        10 -> Games with titled players
        11 -> Games of player (must provide name of playe)
        12 -> Chess (Computer vs Computer, all time controls & ratings)
        13 -> Chess (Computer vs Human, all time controls & ratings)

        **player**
        player name must be at least 3 letters

        **year**
        1999-present

        **month**
        0-12 -> 0 correspond to entire year

        **movetimes**
        0 -> do not include move times
        1 -> include move times

        **download**
        payload variable - must be passed
        """

        
        # Check if the directory exists
        if not os.path.exists(game_dir):
            # Create the directory
            print("Game directory provided doesn't exist.\nCreating new directory")
            os.makedirs(game_dir)
        
        #required headers for scraping
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://www.ficsgames.org/download.html',
            'Origin': 'https://www.ficsgames.org',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
            }

        params = {
            'gametype': game_type,
            'player': player,  
            'year': year,
            'month': month,
            'movetimes': move_times,
            'download': download
        }
        

        try:
            response = requests.post(self.base_url, headers=headers, data=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find the <div> with class "messagetext"
            messagetext_div = soup.find('div', class_='messagetext')
            # Find the <a> tag inside the <div>
            try:
                download_link = messagetext_div.find('a')['href']
            except:
                print("Error: The games you requested do not exist!")
        except requests.RequestException as e:
            print(f"Error connecting to FICS Games Database: {e}")
            return []
        
        full_download_link = "https://www.ficsgames.org" + download_link
        game_bz2 = requests.get(full_download_link, stream=True)

        
        # Check if the request was successful
        if game_bz2.status_code == 200:
            compressed_file_name = download_link.split('/')[-1]
            decompressed_file_name = compressed_file_name.replace('.bz2','')
            decompressed_file_path = f'{game_dir}/{decompressed_file_name}'
            # Save the file in the specified path

            # Open the compressed file and decompress it
            with bz2.BZ2File(filename=game_bz2.raw) as compressed_file:
                # Write the decompressed content to a new file
                with open(decompressed_file_path, 'wb') as decompressed_file:
                    decompressed_file.write(compressed_file.read())
    
            print(f'File downloaded successfully and saved to: {decompressed_file_path}')
        else:
            print(f'Failed to download file. Status code: {game_bz2.status_code}')



# Example usage
if __name__ == "__main__":
    downloader = FICS_Game_Downloader()
    
    games = downloader.download_games(game_type=3,year=2022, month = 2)
    games = downloader.download_games(game_type=4,year=2023, month = 12)

