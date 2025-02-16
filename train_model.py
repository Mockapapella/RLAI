import h5py
import glob
import os
import numpy as np
from typing import Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, optimizer
from torch.amp import autocast, GradScaler
from rl_utils.models import RocketNet
from torch.utils.data import TensorDataset, DataLoader
import gc


"""
- [X] Load data in memory safe way
- [X] Shuffle data that preserves frame:input relationship
- [X] Split training data into an 80/20 split
- [X] Train model
- [X] Validate model
- [X] GOTO shuffle data for N epochs

"""


def get_ram_usage():
    with open("/proc/meminfo", "r") as f:
        lines = f.readlines()

    # Get memory values (they're in kB by default)
    total = int(lines[0].split()[1])
    free = int(lines[1].split()[1])
    available = int(lines[2].split()[1])

    # Convert to GB
    total_gb = total / (1024 * 1024)
    free_gb = free / (1024 * 1024)
    used_gb = total_gb - (available / (1024 * 1024))
    used_percent = (used_gb / total_gb) * 100

    print(f"Total RAM: {total_gb:.2f} GB")
    print(f"Used RAM: {used_gb:.2f} GB")
    print(f"Free RAM: {free_gb:.2f} GB")
    print(f"RAM Usage: {used_percent:.1f}%")


def batch_generator(h5_files):
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            yield f["frames"][:], f["inputs"][:]


def get_test_data(frames, inputs):
    X, y = frames[: round(len(frames) * 0.8)], inputs[: round(len(frames) * 0.8)]
    X_test, y_test = (
        frames[round(len(frames) * 0.8) :],
        inputs[round(len(frames) * 0.8) :],
    )

    return X, y, X_test, y_test


pattern = os.path.join("data/rocket_league/training/", "*_batch.h5")
h5_files = glob.glob(pattern)
# print("=" * 10 + " STARTING TRAINING " + "=" * 10)
# get_ram_usage()

# Training configuration
model = RocketNet()
device = torch.device("cuda")
model.to(device)
# Lower learning rate and add weight decay for better stability
optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_function = nn.BCEWithLogitsLoss()
epochs = 2
batch_size = 8
accumulation_steps = 4  # Gradient accumulation steps
scaler = GradScaler("cuda")  # For mixed precision training
max_grad_norm = 1.0  # For gradient clipping

# DataLoader configuration
dataloader_kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": True,
}

print("=" * 10 + " STARTING TRAINING " + "=" * 10)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0

    for num, (frames, inputs) in enumerate(batch_generator(h5_files)):
        X, y, X_test, y_test = get_test_data(frames, inputs)

        # Create datasets without moving to GPU
        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)

        # Create validation dataset with larger batch size
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, pin_memory=True
        )

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Mixed precision training with proper context manager
            with autocast("cuda"):
                outputs = model(batch_X)
                loss = loss_function(outputs, batch_y)
                loss = (
                    loss / accumulation_steps
                )  # Normalize loss for gradient accumulation

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_count}. Skipping batch.")
                    optim.zero_grad()
                    continue

                # Gradient accumulation and scaling
                scaler.scale(loss).backward()

                if (batch_count + 1) % accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optim)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # Step optimizer and scaler
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"Epoch: {epoch}, batch: {batch_count}, loss: {loss.item():.4f}")

            # Memory cleanup
            del batch_X, batch_y, outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0

            for test_X, test_y in test_loader:
                test_X = test_X.to(device)
                test_y = test_y.to(device)

                with autocast("cuda"):
                    test_outputs = model(test_X)
                    test_loss = loss_function(test_outputs, test_y)

                val_loss += test_loss.item()
                val_batches += 1

                # Memory cleanup
                del test_X, test_y, test_outputs
                torch.cuda.empty_cache()

            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")

        model.train()

    # Print epoch summary
    avg_epoch_loss = epoch_loss / batch_count
    print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

    # Freeze input reduction layer after first epoch to save memory
    if epoch == 0:
        model.freeze_input_reduction()
        print("Input reduction layer frozen for memory optimization")

# Save the model
torch.save(model.state_dict(), "rocket_model.pth")
print("Model saved successfully")
