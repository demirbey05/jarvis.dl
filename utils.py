import requests
from pathlib import Path
import tqdm

def download_if_not_exists(url: str, filepath: str) -> bool:
    """
    Download a file from a URL if it doesn't already exist.
    
    Args:
        url (str): URL of the file to download
        filepath (str): Local path where to save the file
        
    Returns:
        bool: True if file was downloaded or already exists, False if download failed
    """
    # Convert to Path object for better path handling
    filepath = Path(filepath)
    
    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if filepath.exists():
        print(f"File already exists: {filepath}")
        return True
    
    try:
        print(f"Downloading {url} to {filepath}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Get total file size in bytes
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filepath, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                
        print(f"Successfully downloaded to {filepath}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partially downloaded file if it exists
        if filepath.exists():
            filepath.unlink()
        return False
