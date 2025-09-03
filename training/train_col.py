from ..src.colOT.src.col import *
from ..src.prerpoc import *
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset, Dataset

'''
class BatchedFileDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_file, data_dir, device="cpu"):
        """
        Args:
            manifest_file (str): Path to the manifest file listing all batch files.
            data_dir (str): Directory where the batch files are stored.
            device (str): Device to load the data onto ("cpu" or "cuda").
        """
        self.data_dir = data_dir
        self.device = device
        with open(manifest_file, "r") as f:
            self.batch_files = [line.strip() for line in f]
        
        # Load metadata: number of samples per file
        self.file_sample_counts = []
        for batch_file in self.batch_files:
            data = torch.load(os.path.join(data_dir, batch_file), map_location="cpu")
            self.file_sample_counts.append(data.shape[0])
        
        # Calculate global indices for samples
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.file_sample_counts), dim=0)

    def __len__(self):
        """Total number of samples across all files."""
        return self.cumulative_sizes[-1].item()
    
    def __getitem__(self, idx):
        """
        Retrieves the sample corresponding to the global index `idx`.
        """
        # Determine which file contains the sample at `idx`
        file_idx = torch.searchsorted(self.cumulative_sizes, idx, right=True).item()
        
        # Load the corresponding file
        batch_path = os.path.join(self.data_dir, self.batch_files[file_idx])
        data = torch.load(batch_path, map_location="cpu", weights_only=True).to(self.device)

        # Determine the sample's local index within the file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1].item()
        
        # Return the corresponding sample
        return data[local_idx]
'''

'''
class PreloadedBatchedFileDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_file, data_dir, device="cpu"):
        """
        Preload all data files into memory for faster access.
        """
        self.data = []
        self.device = device
        with open(manifest_file, "r") as f:
            batch_files = [line.strip() for line in f]
        
        for batch_file in batch_files:
            batch_path = os.path.join(data_dir, batch_file)
            data = torch.load(batch_path, map_location="cpu")  # Load to CPU initially
            self.data.append(data.to(self.device))  # Move to the desired device if needed

        # Flatten data into a single list of samples
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
'''

'''
class PreloadedBatchedFileDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_file, data_dir, device="cpu"):
        """
        Preload all data files into CPU memory for faster access. Data is moved to GPU during batching.
        """
        self.data = []
        self.device = device
        with open(manifest_file, "r") as f:
            batch_files = [line.strip() for line in f]
        
        # Load all data into CPU memory
        for batch_file in batch_files:
            batch_path = os.path.join(data_dir, batch_file)
            data = torch.load(batch_path, map_location="cpu")  # Load to CPU only
            self.data.append(data)

        # Flatten data into a single list of samples
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Only move the required sample to the GPU
        return self.data[idx].to(self.device)
'''

class EfficientBatchedFileDataset(Dataset):
    def __init__(self, manifest_file, data_dir, device="cpu"):
        """
        Efficiently load data from files on demand to avoid memory overflow.
        Args:
            manifest_file (str): Path to the manifest file listing all batch files.
            data_dir (str): Directory where the batch files are stored.
            device (str): Device to load the data onto ("cpu" or "cuda").
        """
        self.data_dir = data_dir
        self.device = device
        
        # Read the manifest file that lists all batch files
        with open(manifest_file, "r") as f:
            self.batch_files = [line.strip() for line in f]

        # Precompute cumulative sizes of the batches for efficient index lookup
        self.file_sample_counts = []
        for batch_file in self.batch_files:
            batch_path = os.path.join(self.data_dir, batch_file)
            data = torch.load(batch_path, map_location="cpu")  # Load on CPU first
            self.file_sample_counts.append(data.shape[0])  # Record the number of samples in each batch

        # Cumulative sizes to help in identifying which batch the sample belongs to
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.file_sample_counts), dim=0)

    def __len__(self):
        """Return total number of samples across all batch files."""
        return self.cumulative_sizes[-1].item()

    def __getitem__(self, idx):
        """Retrieve a sample corresponding to global index `idx`."""
        # Find the index of the batch file that contains the sample
        file_idx = torch.searchsorted(self.cumulative_sizes, idx, right=True).item()
        
        # Determine the path to the batch file
        batch_path = os.path.join(self.data_dir, self.batch_files[file_idx])
        
        # Load the batch file into memory only when needed
        data = torch.load(batch_path, map_location="cpu")  # Load batch into CPU memory
        sample_count_in_file = data.shape[0]
        
        # Find the local index of the sample within the file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1].item()

        # Return the sample, moving it to the appropriate device
        return data[local_idx].to(self.device)  # Move the sample to the GPU or keep it on CPU


def training_loop(model, loader, n_epochs, optim, device, store_path="col5_model.pt", sort_every=5, reset=False):
    mse = nn.MSELoss()

    if reset is True:
        state_dict = torch.load(store_path,map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("model loaded")
        hist_loss = np.load("hist_loss_col5_Np"+str(len(loader.dataset))+".npy").tolist()
        best_loss = np.min(hist_loss)
    else:
        hist_loss = []
        best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            optim.zero_grad()

            temp = batch.to(device) #(item.to(device) for item in batch)
            nch = int(temp.shape[1]/2)
            b = temp.shape[0]

            #print(temp.shape, flush=True)

            x0 = temp[:,0:nch,:]
            z0 = temp[:,nch:,:]
            #x0 = temp[:,0,...]
            #z0 = temp[:,1,...]

            # Getting model estimation of noise based on the images and the time-step
            #t = torch.randint(0, model.num_timesteps, (b,), device=device).long()
            #t = torch.randint(0, model.num_timesteps, (1,), device=device).long()*torch.ones((b,), device=device).long()
            #t =  torch.randint(0, model.num_timesteps, (1,), device=device)*torch.ones((b,), device=device).long()
            t =  torch.randint(0, model.num_timesteps, (b,), device=device).long()
            w0 = t/model.num_timesteps
            w1 = (t+1)/model.num_timesteps
            w0 = w0.view(-1, 1, 1, 1)
            w1 = w1.view(-1, 1, 1, 1)
            x = col.backward(w0*x0 + (1.-w0)*z0, t)

            # Optimizing the MSE between the noise plugged and the predicted noise
            #w = t/model.num_timesteps
            #w = w.view(-1, 1, 1, 1)
            #loss = mse(x, w*x0 + (1.-w)*z0)

            loss = mse(x, w1*x0 + (1.-w1)*z0)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.4f}"

        hist_loss.append(epoch_loss)
        np.save("hist_loss_col5_Np"+str(len(loader.dataset))+".npy", hist_loss)

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string, flush=True)
        torch.cuda.empty_cache()

print("Start!", flush=True)

device = "cuda"
#device = "cpu"
reset = False
lr = 8e-5
n_epochs = 100
n_files = 10000
train_batch_size = 10
num_timesteps = 10

# read data
image_size = [184,224]
folder = './img_align_celeba/'

data_dir = "./pre_data"
manifest_file = os.path.join(data_dir, "manifest.txt")
#dataset = PreloadedBatchedFileDataset(manifest_file, data_dir, device=device)
dataset = EfficientBatchedFileDataset(manifest_file, data_dir, device=device)
loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

model = Unet(
    dim=64,                  # Base dimension for the model
    channels=3,             # Number of input channels (RGB)
    dim_mults=(1, 2, 4, 8), # Downsampling multipliers
    flash_attn = True,
    learned_variance=False, # Output single channel per pixel
)
print("Unet instantiated", flush=True)

col = MyCOL(model, device=device, num_timesteps=num_timesteps)
print("col instantiated", flush=True)

optimizer = optim.Adam(col.parameters(), lr=lr)
print("optimizer set", flush=True)

training_loop(col, loader, n_epochs, optimizer, device=device, store_path="col5_model_Np"+str(n_files)+".pt", reset=reset)

del dl, dataset_gpu
torch.cuda.empty_cache()

col.eval()

######
nrows, ncols = 5, 6
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))  # Adjust figsize as needed

k=0
for i in range(nrows):
    for j in range(ncols):
        x = torch.randn((1, 3, image_size[0], image_size[1])).to(device)
        #t = num_timesteps*torch.ones((1,), device=device).long()
        x = col.sample(x)

        x = x.detach().cpu().numpy()[0, ...].T
        x = np.swapaxes(x,0,1)
        ax[i, j].imshow(x)
        ax[i, j].axis('off')  # Turn off ticks and labels
        k += 1

        del x
        torch.cuda.empty_cache()

# Remove extra space between subplots
plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig("col5_celeba_nfiles_"+str(n_files)+".pdf", dpi=300, bbox_inches='tight', pad_inches=0)
