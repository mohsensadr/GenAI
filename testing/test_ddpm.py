from ddpm import *
from prerpoc import *
from matplotlib import pyplot as plt

Ndata = 15000

model = Unet(
    dim=64,                 # Base dimension for the model
    channels=3,             # Number of input channels (RGB)
    dim_mults=(1, 2, 4, 8), # Downsampling multipliers
    flash_attn = True,
    learned_variance=False, # Output single channel per pixel
)
print("Unet instantiated")

diffusion = GaussianDiffusion(
    model,
    image_size = (184,224),
    timesteps = 100    # number of steps
)
print("diffusion instantiated")

store_path="ddpm_model_Np"+str(Ndata)+".pt"
state_dict = torch.load(store_path,map_location=torch.device('cuda'))
diffusion.load_state_dict(state_dict)

diffusion.eval()

#sampled_images = diffusion.sample(batch_size = 2).to("cpu")
#plt.matshow(sampled_images[0,...].T)

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

plt.savefig("ddpm_celeba_Np"+str(Ndata)+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
