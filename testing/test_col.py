import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.colOT.col import *
from src.prerpoc import *
from matplotlib import pyplot as plt
from huggingface_hub import hf_hub_download

device = "cpu"

#name_dataset = "mnist"
#name_dataset = "CIFAR10"
name_dataset = "Food101"

repo_id = "mohsensadr91/"+name_dataset     # your repo name on Hugging Face
filename = f"col_{name_dataset}.pt"  # must match the filename you uploaded

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

image_size = params["image_size"]
num_channel = params["channels"]
col = MyCOL(model, device=device, num_timesteps=params["timesteps"])

col.load_state_dict(checkpoint["model_state"])

col.eval()

#################
fig, ax = plt.subplots(figsize=(5, 4))  # Adjust figsize as needed
ax.plot(checkpoint["hist_loss"], marker='o', markersize=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title(f"Training Loss for COL on {name_dataset}")
plt.savefig("col_"+name_dataset+"_loss.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

nrows, ncols = 5, 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 4))  # Adjust figsize as needed

k=0
for i in range(nrows):
    for j in range(ncols):
        x = torch.randn((1, num_channel, image_size[0], image_size[1])).to(device)
        x = torch.clamp((col.sample(x) + 1) / 2, 0, 1)

        x = x.detach().cpu().numpy()[0, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

plt.subplots_adjust(wspace=0, hspace=0.2)
fig.savefig("col_"+name_dataset+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
