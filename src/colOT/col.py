from ..libs import *

class MyCOL(nn.Module):
    def __init__(self, network, device=None, num_timesteps=10):
        super(MyCOL, self).__init__()
        self.device = device
        self.network = network.to(device)
        self.num_timesteps = num_timesteps
    def OT_col(self, x, y, avg_window=10, tol=1e-6, MinIter=10, MaxIter=100):
      tries = int(len(x))
      sum_ = 1000000
      dists_coll = torch.zeros(MaxIter+1, device=x.device)
      dists_coll[0] = torch.mean((x - y)**2)
      for nt in range(MaxIter):
        iss = torch.randperm(len(x), device=x.device)
        i1s = iss[:int(tries/2)]
        i2s = iss[int(tries/2):]

        # Calculate the initial distances
        s0 = torch.sum( (x[i1s] - y[i1s])**2, axis=-1) + torch.sum( (x[i2s] - y[i2s])**2, axis=-1)

        # Calculate the distances after the swap
        s1 = torch.sum( (x[i1s] - y[i2s])**2, axis=-1) + torch.sum( (x[i2s] - y[i1s])**2, axis=-1)

        # Determine which swaps to accept
        mask = s1 < s0

        # Perform the swaps for accepted cases
        accepted_i1s = i1s[mask]
        accepted_i2s = i2s[mask]

        #y[accepted_i1s], y[accepted_i2s] = y[accepted_i2s], y[accepted_i1s]

       # Use in-place swap to ensure y remains on the correct device
        y_clone = y.clone()  # Workaround for in-place swapping with PyTorch
        y_clone[accepted_i1s], y_clone[accepted_i2s] = y_clone[accepted_i2s], y_clone[accepted_i1s]
        y = y_clone  # Update y after swapping

        dists_coll[nt+1] = torch.mean((x - y)**2)
        if nt>avg_window and nt > MinIter:
          sum_0 = torch.sum(dists_coll[-avg_window:])
          if abs(sum_ - sum_0)/sum_0 < tol:
            break
          sum_ = sum_0
      return x, y

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        #return self.network(x, torch.ones(1, device=x.device))
        return self.network(x, t)

    def sample(self, x):
        with torch.no_grad():
            b = x.shape[0]
            t =  torch.zeros((b,), device=x.device).long()
            for i in range(self.num_timesteps):
                # Convert ti to a tensor to maintain consistency with network input
                x = self.network(x, t)
                t = t + 1
            return x

def OT_col(x, y, avg_window=10, tol=1e-6, MinIter=10, MaxIter=1000):
      tries = int(len(x))
      sum_ = 1000000
      dists_coll = torch.zeros(MaxIter+1, device=x.device)
      dists_coll[0] = torch.mean(torch.sum((x - y)**2, axis=-1))
      iter = 0

      for nt in range(MaxIter):
        iss = torch.randperm(len(x), device=x.device)
        i1s = iss[:int(tries/2)]
        i2s = iss[int(tries/2):]

        # Calculate the initial distances
        s0 = torch.sum( (x[i1s,:] - y[i1s,:])**2, axis=-1) + torch.sum( (x[i2s,:] - y[i2s,:])**2, axis=-1)

        # Calculate the distances after the swap
        s1 = torch.sum( (x[i1s,:] - y[i2s,:])**2, axis=-1) + torch.sum( (x[i2s,:] - y[i1s,:])**2, axis=-1)

        # Determine which swaps to accept
        mask = s1 < s0

        # Perform the swaps for accepted cases
        accepted_i1s = i1s[mask]
        accepted_i2s = i2s[mask]

        y[accepted_i1s], y[accepted_i2s] = y[accepted_i2s], y[accepted_i1s]

        # Use in-place swap to ensure y remains on the correct device
        #y_clone = y.clone()  # Workaround for in-place swapping with PyTorch
        #y_clone[accepted_i1s], y_clone[accepted_i2s] = y_clone[accepted_i2s], y_clone[accepted_i1s]
        #y = y_clone  # Update y after swapping

        dists_coll[nt+1] = torch.mean(torch.sum((x - y)**2, axis=-1))
        if nt>avg_window and nt > MinIter:
          sum_0 = torch.sum(dists_coll[-avg_window:])
          if abs(sum_ - sum_0)/sum_0 < tol:
            break
          sum_ = sum_0
        iter += 1
      dists_coll = dists_coll[:iter]
      return x, y, dists_coll
