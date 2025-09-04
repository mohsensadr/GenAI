import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.colOT.col import *
from src.prerpoc import *
from matplotlib import pyplot as plt

#name_dataset = "mnist"
name_dataset = "cifar10"
store_path="../models/col_"+name_dataset+".pt"
device = "cpu"

checkpoint = torch.load(store_path, map_location=torch.device(device))
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
nrows, ncols = 5, 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 4))  # Adjust figsize as needed

k=0
for i in range(nrows):
    for j in range(ncols):
        x = torch.randn((1, num_channel, image_size[0], image_size[1])).to(device)
        x = col.sample(x)

        x = x.detach().cpu().numpy()[0, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

plt.subplots_adjust(wspace=0, hspace=0.2)
fig.savefig("col_"+name_dataset+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
