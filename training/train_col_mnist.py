import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.colOT.col import *
from src.prerpoc import *
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset, Dataset, Subset, TensorDataset, DataLoader
    
def make_ot_dataset(base_dataset, ot_fn, device="cpu", max_samples=None,
                          MinIter=1000, MaxIter=1000, tol=1e-9, avg_window=20):
    imgs = torch.stack([base_dataset[i] for i in range(len(base_dataset))])
    if max_samples is not None:
        imgs = imgs[:max_samples]

    N, C, H, W = imgs.shape
    imgs = imgs.to(device)

    # Flatten to [N, d]
    Xdata_d = imgs.view(N, -1)
    Yt_d = torch.randn_like(Xdata_d)

    # Apply OT across dataset
    Xdata_d, Yt_d, w2_hist = ot_fn(
        Xdata_d, Yt_d, MinIter=MinIter, MaxIter=MaxIter, tol=tol, avg_window=avg_window
    )

    # Reshape and concatenate as [N, 2, H, W]
    Xdata_d = Xdata_d.view(N, C, H, W)
    Yt_d = Yt_d.view(N, C, H, W)
    data_d = torch.cat([Xdata_d, Yt_d], dim=1)

    class OTTensorDataset(torch.utils.data.Dataset):
        def __init__(self, data_tensor):
            self.data_tensor = data_tensor.to(device)
        def __len__(self):
            return len(self.data_tensor)
        def __getitem__(self, idx):
            return self.data_tensor[idx]  # return tensor directly

    dataset = OTTensorDataset(data_d)
    return dataset, w2_hist

def training_loop(checkpoint, model, loader, n_epochs, optim, device, store_path="col_model.pt", resume=False):
    mse = nn.MSELoss()

    # Load checkpoint if resuming
    if resume and os.path.exists(store_path):
        loaded_ckpt = torch.load(store_path, map_location=device)
        model.load_state_dict(loaded_ckpt["model_state"])

        # Only load optimizer if it's in checkpoint
        if loaded_ckpt.get("optimizer_state") is not None:
            optim.load_state_dict(loaded_ckpt["optimizer_state"])

        checkpoint.update({
            "epoch": loaded_ckpt.get("epoch", 0),
            "hist_loss": loaded_ckpt.get("hist_loss", []),
            "best_loss": loaded_ckpt.get("best_loss", float("inf")),
            "model_state": loaded_ckpt["model_state"],
            "optimizer_state": loaded_ckpt.get("optimizer_state", None)
        })
        print(f"Resumed training from epoch {checkpoint['epoch']}, best loss {checkpoint['best_loss']:.4f}")
    
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    hist_loss = checkpoint["hist_loss"]

    for epoch in range(start_epoch, start_epoch+n_epochs):
        epoch_loss = 0.0

        for _, batch in enumerate(loader):
            optim.zero_grad()
            temp = batch.to(device) #(item.to(device) for item in batch)
            nch = int(temp.shape[1]/2)
            b = temp.shape[0]

            x0 = temp[:,0:nch,:]
            z0 = temp[:,nch:,:]
            
            t =  torch.randint(0, model.num_timesteps, (b,), device=device).long()
            w0 = t/model.num_timesteps
            w1 = (t+1)/model.num_timesteps
            w0 = w0.view(-1, 1, 1, 1)
            w1 = w1.view(-1, 1, 1, 1)
            x = col.backward(w0*x0 + (1.-w0)*z0, t)

            loss = mse(x, w1*x0 + (1.-w1)*z0)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

        hist_loss.append(epoch_loss)

        # Update checkpoint runtime info
        checkpoint.update({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "hist_loss": hist_loss,
        })

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, store_path)
            print(f"Epoch {epoch+1}: {epoch_loss:.4f} --> Best model stored!")
        else:
            print(f"Epoch {epoch+1}: {epoch_loss:.4f}")

        torch.cuda.empty_cache()

print("Start!", flush=True)

#device = "cuda"
device = "cpu"
resume = True
lr = 1e-4
n_epochs = 10
train_batch_size = 10
num_workers = 0
timesteps = 10
max_samples = 1000
name_dataset = "MNIST"
image_size = (32, 32)
num_channel = 1 # Number of input channels (grayscale)

dim = 32
dim_mults=(1, 2, 4)
flash_attn = True
learned_variance = False

MinIter=1000
MaxIter=1000

store_path="../models/col_"+name_dataset+".pt"

class ImagesOnly(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # ignore label
        return image

def normalize_minus_one_one(x):
    return x * 2 - 1

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(normalize_minus_one_one)  # Normalize to [-1,1] if needed
])

ds = datasets.MNIST(root='./'+name_dataset, train=True, download=True, transform=transform)
print("data read")

max_size = len(ds)
print("Maximum dataset size:", max_size)

sample_image, _ = ds[0]  # for datasets with (image, label) tuples

print("Type:", type(sample_image))
print("Shape:", sample_image.shape)  # C x H x W for transformed images
print("Height:", sample_image.shape[1])
print("Width:", sample_image.shape[2])
print("Channels:", sample_image.shape[0])

ds_limited = Subset(ds, range(min(max_samples, max_size)))

ds_images_only = ImagesOnly(ds_limited)

ot_dataset, w2_hist = make_ot_dataset(ds_images_only, OT_col, device=device, MinIter=MinIter, MaxIter=MaxIter)

dl = DataLoader(ot_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
print("data loader is constructed")

model = Unet(
    dim=dim,                  # Base dimension for the model
    channels=num_channel,     # Number of input channels (RGB)
    dim_mults=dim_mults,      # Downsampling multipliers
    flash_attn=flash_attn,
    learned_variance=learned_variance,  # Output single channel per pixel
)
print("Unet instantiated", flush=True)

col = MyCOL(model, device=device, num_timesteps=timesteps)
print("col instantiated", flush=True)

optimizer = optim.Adam(col.parameters(), lr=lr)
print("optimizer set", flush=True)

checkpoint = {
    "epoch":0,
    "model_state": col.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "hist_loss": [],
    "best_loss": float("inf"),
    "model_params": {
        "dim": dim,
        "channels": num_channel,
        "dim_mults": dim_mults,
        "flash_attn": flash_attn,
        "learned_variance": learned_variance,
        "image_size": image_size,
        "timesteps": timesteps,
        "device": device
    }
}

training_loop(checkpoint, col, dl, n_epochs, optimizer, device=device, store_path=store_path, resume=resume)

del dl
torch.cuda.empty_cache()

col.eval()

######
nrows, ncols = 5, 6
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))  # Adjust figsize as needed

k=0
for i in range(nrows):
    for j in range(ncols):
        x = torch.randn((1, num_channel, image_size[0], image_size[1])).to(device)
        x = (col.sample(x) + 1 / 2 )

        x = x.detach().cpu().numpy()[0, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

        del x
        torch.cuda.empty_cache()

# Remove extra space between subplots
plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig("col_"+name_dataset+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()