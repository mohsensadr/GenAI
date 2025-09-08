import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.ddpm.ddpm import *
from src.prerpoc import *
from matplotlib import pyplot as plt
from huggingface_hub import hf_hub_download

device = "cpu"

#name_dataset = "mnist"
#name_dataset = "CIFAR10"
name_dataset = "Food101"

repo_id = "mohsensadr91/"+name_dataset     # your repo name on Hugging Face
filename = f"ddpm_{name_dataset}.pt"  # must match the filename you uploaded

# Download checkpoint file from Hub (caches locally after first time)
checkpoint_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="../models",        # 👈 forces saving in your folder
    local_dir_use_symlinks=False  # avoids symlinks to cache
)

checkpoint = torch.load(checkpoint_path, map_location=device)
params = checkpoint["model_params"]

model = Unet(
    dim=params["dim"],
    channels=params["channels"],
    dim_mults=params["dim_mults"],
    flash_attn=params["flash_attn"],
    learned_variance=params["learned_variance"],
)

diffusion = GaussianDiffusion(
    model.to(params["device"]),
    image_size=params["image_size"],
    timesteps=params["timesteps"]
)

diffusion.load_state_dict(checkpoint["model_state"])

diffusion.to(device)

diffusion.eval()

#################
fig, ax = plt.subplots(figsize=(5, 4))  # Adjust figsize as needed
ax.plot(checkpoint["hist_loss"], marker='o', markersize=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title(f"Training Loss for DDPM on {name_dataset}")
plt.savefig("ddpm_"+name_dataset+"_loss.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

nrows, ncols = 5, 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 4))  # Adjust figsize as needed

k=0
for i in range(nrows):
    x0 = diffusion.sample(batch_size = ncols).detach().cpu().numpy()
    for j in range(ncols):
        x = x0[j, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

plt.savefig("ddpm_"+name_dataset+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
