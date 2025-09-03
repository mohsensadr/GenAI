from ..src.ddpm.ddpm import *
from ..src.prerpoc import *
from matplotlib import pyplot as plt

def training_loop(model, loader, n_epochs, optim, device, store_path="ddpm_model.pt", sort_every=5, reset=False):
    mse = nn.MSELoss()

    if reset is True:
        state_dict = torch.load(store_path,map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("model loaded")
        hist_loss = np.load("hist_loss_ddpm_Np"+str(len(loader.dataset))+".npy").tolist()
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

            loss = model(data)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        hist_loss.append(epoch_loss)
        np.save("hist_loss_ddpm_Np"+str(len(loader.dataset))+".npy", hist_loss)

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
        torch.cuda.empty_cache()

device = "cuda"
reset = False
lr = 8e-5
n_epochs = 100
n_files = 15000
train_batch_size = 10
num_workers = 2
timesteps = 100

# read data
image_size = [184,224]
folder = './img_align_celeba/'

ds = CustomDataset(folder, image_size, n_files = n_files, augment_horizontal_flip = True)
print("data read")

dl = DataLoader(ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)
print("loader made")

model = Unet(
    dim=64,                  # Base dimension for the model
    channels=3,             # Number of input channels (RGB)
    dim_mults=(1, 2, 4, 8), # Downsampling multipliers
    flash_attn = True,
    learned_variance=False, # Output single channel per pixel
)
print("Unet instantiated")

diffusion = GaussianDiffusion(
    model.to(device),
    image_size = (184,224),
    timesteps = timesteps    # number of steps
)
diffusion.to(device)
print("diffusion instantiated")

optimizer = optim.Adam(diffusion.parameters(), lr=lr)
print("optimizer set")

torch.cuda.empty_cache()
training_loop(diffusion, dl, n_epochs, optimizer, device=device, store_path="ddpm_model_Np"+str(n_files)+".pt", reset=reset)

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
fig.savefig("ddpm_celeba_nfiles_"+str(n_files)+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)


