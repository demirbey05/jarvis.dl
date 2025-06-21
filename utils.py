import inspect
import collections
import IPython
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
from torch import nn

def add_to_class(Class):
    def wrapper(attribute):
        setattr(Class,attribute.__name__,attribute)
    return wrapper

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        # We will get all arguments of caller function and init
        args = inspect.currentframe().f_back.f_locals
        for k,v in args.items():
            if k not in ignore:
                setattr(self,k,v)


class ProgressBoard(HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)


class Module(nn.Module, HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError


class DataModule(HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    

class Trainer(HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError


# Generated by LLM
def download_file_from_url(url, save_path=None, chunk_size=8192):
    """
    Download a file from a URL to a local path with progress bar.
    
    Args:
        url (str): URL of the file to download
        save_path (str, optional): Path where to save the downloaded file. 
                                 If None, uses the filename from the URL.
        chunk_size (int, optional): Size of chunks to download at a time. Defaults to 8KB.
    
    Returns:
        str: Absolute path where the file was saved
    
    Raises:
        Exception: If there's an error during download or file saving
    """
    import os
    import requests
    from tqdm import tqdm
    from urllib.parse import urlparse
    
    try:
        # Send HTTP GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get the filename from URL if save_path is not provided
        if save_path is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = 'downloaded_file'
            save_path = filename
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Get the total file size from headers if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file with progress bar
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return os.path.abspath(save_path)
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download file from {url}: {str(e)}")
    except IOError as e:
        raise Exception(f"Failed to save file to {save_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")    

# Generated by LLM
def extract_zip(zip_path, extract_to=None, remove_zip=False):
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_to (str, optional): Directory to extract to. If None, extracts to a folder
                                   with the same name as the ZIP file (without .zip extension)
        remove_zip (bool, optional): Whether to remove the ZIP file after extraction.
                                     Defaults to False.
    
    Returns:
        str: Path to the directory where files were extracted
    
    Raises:
        FileNotFoundError: If the ZIP file doesn't exist
        zipfile.BadZipFile: If the file is not a valid ZIP file
        Exception: For other errors during extraction
    """
    import os
    import zipfile
    from tqdm import tqdm

    # Check if file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Set default extract directory if not provided
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]
    
    # Create target directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # Open the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to extract
            file_list = zip_ref.namelist()
            
            # Extract with progress bar
            for file in tqdm(file_list, desc=f"Extracting {os.path.basename(zip_path)}", unit="files"):
                try:
                    zip_ref.extract(file, extract_to)
                except Exception as e:
                    print(f"Warning: Could not extract {file}: {str(e)}")
            
            # Get the actual extracted directory (in case of a single root directory in zip)
            extracted_dirs = [f for f in os.listdir(extract_to) 
                           if os.path.isdir(os.path.join(extract_to, f))]
            
            # If there's only one directory and it's the same as the zip's name, return that
            if len(extracted_dirs) == 1:
                actual_extract = os.path.join(extract_to, extracted_dirs[0])
            else:
                actual_extract = extract_to
        
        # Remove the zip file if requested
        if remove_zip:
            os.remove(zip_path)
            
        return os.path.abspath(actual_extract)
        
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"File is not a valid ZIP file: {zip_path}")
    except Exception as e:
        # Clean up partially extracted files if there was an error
        if os.path.exists(extract_to):
            import shutil
            shutil.rmtree(extract_to, ignore_errors=True)
        raise Exception(f"Failed to extract {zip_path}: {str(e)}")




def show_attention_heatmaps(matrices,xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='Reds'):

    use_svg_display()
    row_numbers,col_numbers,_,_ = matrices.shape # matrices shape must be (row_numbers,col_numbers,keys,queries)
    fig,axes = plt.subplots(row_numbers,col_numbers,figsize=figsize, sharex=True, sharey=True, squeeze=False)

    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
        for j, (ax,mat) in enumerate(zip(row_axes,row_matrices)):
            pcm = ax.imshow(mat.detach().numpy(), cmap=cmap)
            if i==row_numbers-1:
                ax.set_xlabel(xlabel)
            if j==0:
                ax.set_ylabel(ylabel)
            if titles is not None:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)        
                
            
    
