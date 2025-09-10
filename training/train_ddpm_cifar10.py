import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.ddpm.ddpm import *
from src.prerpoc import *
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset

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

def training_loop(checkpoint, model, loader, n_epochs, optim, device,
                  store_path="../models/ddpm_model.pt", resume=False):
    mse = nn.MSELoss()

    # Resume if checkpoint exists
    if resume and os.path.exists(store_path):
        loaded_ckpt = torch.load(store_path, map_location=device)
        model.load_state_dict(loaded_ckpt["model_state"])
        if loaded_ckpt.get("optimizer_state") is not None:
            optim.load_state_dict(loaded_ckpt["optimizer_state"])

        checkpoint.update({
            "epoch": loaded_ckpt.get("epoch", 0),
            "hist_loss": loaded_ckpt.get("hist_loss", []),
            "best_loss": loaded_ckpt.get("best_loss", float("inf")),
            "model_state": loaded_ckpt["model_state"],
            "optimizer_state": loaded_ckpt.get("optimizer_state", None),
        })
        print(f"Resumed training from epoch {checkpoint['epoch']}, "
              f"best loss {checkpoint['best_loss']:.4f}")
    else:
        checkpoint.update({
            "epoch": 0,
            "hist_loss": [],
            "best_loss": float("inf"),
        })

    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    hist_loss = checkpoint["hist_loss"]

    #for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
    for epoch in range(start_epoch, start_epoch + n_epochs):
        epoch_loss = 0.0
        #for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
        for step, batch in enumerate(loader):
            optim.zero_grad()

            data = batch.to(device)

            if step == 0 and epoch == 0:
              print(f"[Debug] First batch device: {data.device}")

            loss = model(data)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

        hist_loss.append(epoch_loss)
        # Update checkpoint
        checkpoint.update({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "hist_loss": hist_loss,
        })

        # Save if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, store_path)
            print(f"Epoch {epoch+1}: {epoch_loss:.4f} --> Best model stored!")
        else:
            print(f"Epoch {epoch+1}: {epoch_loss:.4f}")

        torch.cuda.empty_cache()

def main():
    device = "cuda"
    #device = "cpu"
    resume = False
    lr = 1e-4
    n_epochs = 10
    train_batch_size = 10
    num_workers = 2
    timesteps = 10
    max_samples = 50000
    name_dataset = "CIFAR10"
    image_size = (32, 32)
    num_channel = 3 # Number of input channels (RGB)

    dim = 32
    dim_mults=(1, 2, 4, 8)
    flash_attn = True
    learned_variance = False

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize_minus_one_one)  # Normalize to [-1,1] if needed
    ])

    ds = datasets.CIFAR10(root='./'+name_dataset, train=True, download=True, transform=transform)
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
    dl = DataLoader(ds_images_only, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    print("data loader is constructed")

    model = Unet(
        dim=dim,                  # Base dimension for the model
        channels=num_channel,    # Number of input channels (RGB)
        dim_mults=dim_mults,     # Downsampling multipliers
        flash_attn = flash_attn,
        learned_variance=learned_variance, # Output single channel per pixel
    )
    print("Unet instantiated")

    diffusion = GaussianDiffusion(
        model.to(device),
        image_size = image_size, # Image dimensions
        timesteps = timesteps    # number of steps
    )

    diffusion.to(device)
    print("diffusion instantiated")

    optimizer = optim.Adam(diffusion.parameters(), lr=lr)
    print("optimizer set")

    checkpoint = {
        "epoch":0,
        "model_state": diffusion.state_dict(),
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

    torch.cuda.empty_cache()
    training_loop(checkpoint, diffusion, dl, n_epochs, optimizer, device=device, store_path="../models/ddpm_"+name_dataset+".pt", resume=resume)

    diffusion.eval()

    ######
    nrows, ncols = 5, 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))  # Adjust figsize as needed

    k=0
    for i in range(nrows):
        for j in range(ncols):
            x = diffusion.sample(batch_size = 1)
            x = x.detach().cpu().numpy()[0, ...].T
            x = np.swapaxes(x,0,1)
            ax[i, j].imshow(x)
            ax[i, j].axis('off')  # Turn off ticks and labels
            k += 1

    # Remove extra space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("ddpm_"+name_dataset+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()