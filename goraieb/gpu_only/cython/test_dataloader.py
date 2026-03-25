# test_dataloader.py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Fake a small mmap dataset
X = np.random.randn(1000, 773).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)
np.save("/tmp/test_X.npy", X)
np.save("/tmp/test_y.npy", y)

# Import your MmapDataset
from train_nn_gpu import MmapDataset, ChessEvaluator

# Write as mmap
Xm = np.memmap("/tmp/test_X.dat", dtype=np.float32, mode='w+', shape=(1000, 773))
ym = np.memmap("/tmp/test_y.dat", dtype=np.float32, mode='w+', shape=(1000,))
Xm[:] = X; ym[:] = y
Xm.flush(); ym.flush()

indices = list(range(1000))
ds = MmapDataset("/tmp/test_X.dat", "/tmp/test_y.dat", 1000, indices)

loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

print("Testing DataLoader...")
for i, (xb, yb) in enumerate(loader):
    if i == 0:
        print(f"  batch shape: X={xb.shape}, y={yb.shape}")
print(f"  {i+1} batches OK ✓")

# Quick training smoke test
model = ChessEvaluator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()
xb, yb = next(iter(loader))
out = model(xb.to(device))
loss = torch.nn.SmoothL1Loss()(out, yb.to(device))
loss.backward()
print(f"  forward + backward OK ✓ (loss={loss.item():.4f})")
print("All good.")
