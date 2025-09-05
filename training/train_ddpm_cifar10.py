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

def training_loop(checkpoint, model, loader, n_epochs, optim, device, store_path="../model/ddpm_model.pt", sort_every=5, reset=False, name_dataset=""):
    mse = nn.MSELoss()

    if reset is True:
        state_dict = torch.load(store_path,map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("model loaded")
        hist_loss = np.load("hist_loss_ddpm_"+name_dataset+".npy").tolist()
        best_loss = np.min(hist_loss)
    else:
        best_loss = float("inf")
        hist_loss = []

    #for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
    for epoch in range(n_epochs):
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

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        hist_loss.append(epoch_loss)
        np.save("hist_loss_ddpm_"+name_dataset+".npy", hist_loss)

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
        torch.cuda.empty_cache()

device = "cuda" #"cpu"
reset = False
lr = 8e-5
n_epochs = 10
train_batch_size = 10
num_workers = 2
timesteps = 10
max_samples = 10000
name_dataset = "CIFAR10"
image_size = (32, 32)
num_channel = 3 # Number of input channels (RGB)

dim = 64
dim_mults=(1, 2, 4, 8)
flash_attn = True
learned_variance = False

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

ds = datasets.CIFAR10(root='./'+name_dataset, train=True, download=True, transform=transform)
print("data read")

ds_limited = Subset(ds, range(max_samples))
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

checkpoint = {
    "model_state": diffusion.state_dict(),
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

diffusion.to(device)
print("diffusion instantiated")

optimizer = optim.Adam(diffusion.parameters(), lr=lr)
print("optimizer set")

torch.cuda.empty_cache()
training_loop(checkpoint, diffusion, dl, n_epochs, optimizer, device=device, store_path="../models/ddpm_"+name_dataset+".pt", reset=reset, name_dataset=name_dataset)

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
