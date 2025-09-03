from torchvision.io import read_image
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
from torchvision import transforms as T, utils
from PIL import Image
from colOT.col import *

print("libs loaded", flush=True)

class CustomDataset(TorchDataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        n_files=None,  # Number of files to read
        offset=0,      # Offset for slicing the file paths
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # Apply the offset and limit the number of files if n_files is specified
        if n_files is not None:
            self.paths = self.paths[offset:offset + n_files]

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

def process_and_save_batch(batch, output_dir, batch_idx):
    output_path = os.path.join(output_dir, f"batch_{batch_idx}.pt")
    torch.save(batch.to("cpu"), output_path)
    print(f"Saved batch {batch_idx} to {output_path}", flush=True)

def save_manifest(output_dir, num_batches):
    with open(os.path.join(output_dir, "manifest.txt"), "w") as f:
        for i in range(num_batches):
            f.write(f"batch_{i}.pt\n")

device = "cuda"
#device = "cpu"
image_size = [184,224]
folder = './img_align_celeba/'
images_per_batch = 1000
total_images = 100000 #len(Path(folder).rglob('*'))
output_dir = "./pre_data/"
os.makedirs(output_dir, exist_ok=True)

batch_idx = 0
offset = 0

while offset < total_images:
    # Limit files for the current batch
    n_files = min(images_per_batch, total_images - offset)
    
    # Create dataset with offset
    ds = CustomDataset(folder, image_size, n_files=n_files, offset=offset, augment_horizontal_flip=True)

    Xdata_d = torch.empty((len(ds), 3, image_size[0], image_size[1]), dtype=torch.float, device=device)
    for i, img in enumerate(ds):
        Xdata_d[i,...] = img.to(device)
    del ds

    imsh = Xdata_d.shape[1:]
    Ndata = Xdata_d.shape[0]
    Xdata_d = Xdata_d.reshape(Ndata,imsh[0]*imsh[1]*imsh[2])
    Yt_d = torch.randn(Xdata_d.shape).to(device)

    Xdata_d, Yt_d, w2_hist = OT_col(Xdata_d, Yt_d, MinIter=1000, MaxIter=1000, tol = 1e-9, avg_window=20)

    data_d = torch.concatenate([Xdata_d[None,...],Yt_d[None,...]], axis=0)
    del Xdata_d, Yt_d, w2_hist
    torch.cuda.empty_cache()
    data_d = torch.swapaxes(data_d,0,1).reshape(Ndata, 2*imsh[0], imsh[1], imsh[2])

    process_and_save_batch(data_d, output_dir, batch_idx)

    save_manifest(output_dir, batch_idx)

    del data_d
    torch.cuda.empty_cache()

    batch_idx += 1
    offset += n_files
