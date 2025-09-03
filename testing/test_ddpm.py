import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.ddpm.ddpm import *
from src.prerpoc import *
from matplotlib import pyplot as plt

store_path="../models/ddpm_mnist.pt"
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

diffusion = GaussianDiffusion(
    model.to(params["device"]),
    image_size=params["image_size"],
    timesteps=params["timesteps"]
)

diffusion.load_state_dict(checkpoint["model_state"])

diffusion.eval()

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

plt.savefig("ddpm_mnist.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
