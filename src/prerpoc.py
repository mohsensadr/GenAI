from libs import *

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset 
from pathlib import Path
from torchvision import transforms as T, utils
from multiprocessing import cpu_count
from torch.optim import Adam
from PIL import Image

class CustomDataset(TorchDataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        n_files=None,  # New parameter for the number of files to read
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # Limit the number of files if n_files is specified
        if n_files is not None:
            self.paths = self.paths[:n_files]

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if convert_image_to is not None
            else nn.Identity()
        )

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
