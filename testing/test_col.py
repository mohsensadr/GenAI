from col import *
from prerpoc import *
from matplotlib import pyplot as plt

n_files = 40000

model = Unet(
    dim=64,                  # Base dimension for the model
    channels=3,             # Number of input channels (RGB)
    dim_mults=(1, 2, 4, 8), # Downsampling multipliers
    flash_attn = True,
    learned_variance=False, # Output single channel per pixel
)

print("Unet instantiated")
num_timesteps = 10
image_size = [184,224]
device = "cuda"
col = MyCOL(model, device=device, num_timesteps=num_timesteps)
store_path="col5_model_Np"+str(n_files)+".pt"
state_dict = torch.load(store_path,map_location=torch.device(device))
col.load_state_dict(state_dict)
col.eval()

#################
nrows, ncols = 5, 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 4))  # Adjust figsize as needed

k=0
for i in range(nrows):
    for j in range(ncols):
        x = torch.randn((1, 3, image_size[0], image_size[1])).to(device)
        x = col.sample(x)

        x = x.detach().cpu().numpy()[0, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig("col5_celeba_nfiles_"+str(n_files)+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
