import torch


class GaussianNoise(torch.nn.Module):
    """Add gaussian noise with a given std"""
    def __init__(self, sig=1.):
        super().__init__()
        self.sig = sig

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = 1
        return x + torch.randn(shape, device=x.device) * self.sig


class MemBatch:
    """Store batches of data in a library and sample examples as well as pure noise examples"""
    def __init__(self, capacity, shape, *, device, noise_capacity=0):
        self.mem = torch.zeros((capacity + noise_capacity, *shape), device=device).normal_(0, 1)
        self.device = device
        self.capacity = capacity
        self.noise_capacity = noise_capacity
        self.head = 0
        self.full = False

    def add(self, stacked_entries):
        if stacked_entries.shape[0] > self.capacity:
            print("trying to save more data than capacity: shaving off the end of the batch")
            entries = stacked_entries[:self.capacity].detach()
        else:
            entries = stacked_entries.detach()

        if self.head + entries.shape[0] <= self.capacity:
            self.mem[self.head:self.head + entries.shape[0]] = entries
        else:
            self.mem[self.head:self.capacity] = entries[:self.capacity - self.head]
            self.mem[:entries.shape[0] - (self.capacity - self.head)] = entries[self.capacity - self.head:]
            self.full = True

        self.head = (self.head + entries.shape[0]) % self.capacity

    def sample(self, batch_size):
        if self.full:
            indices = torch.randint(self.capacity + self.noise_capacity, (batch_size,))
        else:
            indices = torch.randint(max(self.head, 1), (batch_size,))
        return self.mem[indices]

    def add_k_best(self, data, scores, k=16):
        indices = torch.argsort(scores)[:k]
        self.add(data[indices].to(self.device))