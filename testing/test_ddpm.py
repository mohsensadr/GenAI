import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.ddpm.ddpm import *
from src.prerpoc import *
from matplotlib import pyplot as plt

device = "cpu"
image_size = (32, 32)
num_channel = 1
timesteps = 10
store_path="../models/ddpm_mnist.pt"

model = Unet(
    dim=16,                  # Base dimension for the model
    channels=num_channel,    # Number of input channels (RGB)
    dim_mults=(1, 2, 4),     # Downsampling multipliers
    flash_attn = True,
    learned_variance=False, # Output single channel per pixel
)
print("Unet instantiated")

diffusion = GaussianDiffusion(
    model.to(device),
    image_size = image_size, # Image dimensions
    timesteps = timesteps    # number of steps
)
print("diffusion instantiated")

state_dict = torch.load(store_path,map_location=torch.device(device))
diffusion.load_state_dict(state_dict)

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
