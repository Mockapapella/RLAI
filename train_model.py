import h5py
import glob
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from rl_utils.models import RocketNet
from torch.utils.data import TensorDataset, DataLoader
import gc
from time import time
from datetime import datetime, timedelta


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def format_percentage(current, total):
    return f"{(current / total * 100):.1f}%"


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


def split_batch(frames, inputs, train_ratio=0.8, max_samples=5000):
    """Split a batch into train/val sets with optional subsampling"""
    # Determine total samples to use
    n_samples = min(len(frames), max_samples) if max_samples else len(frames)

    # Generate random indices for the full dataset
    indices = np.random.permutation(len(frames))[:n_samples]

    # Calculate split point
    split_idx = int(n_samples * train_ratio)

    # Split indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Return split data
    return (
        frames[train_indices],
        inputs[train_indices],
        frames[val_indices],
        inputs[val_indices],
    )


def batch_generator(h5_files):
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            frames, inputs = f["frames"][:], f["inputs"][:]
            yield h5_file, split_batch(frames, inputs)


# Get files
pattern = os.path.join("data/rocket_league/training/", "*_batch.h5")
h5_files = glob.glob(pattern)

# Set random seed for reproducibility
random.seed(42)

# Training configuration
model = RocketNet()
device = torch.device("cuda")
model.to(device)
# Lower learning rate and add weight decay for better stability
optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_function = nn.BCEWithLogitsLoss()
epochs = 5
batch_size = 40
accumulation_steps = 1  # Gradient accumulation steps
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

# Log training configuration
print("\n" + "=" * 50)
print("TRAINING CONFIGURATION")
print("=" * 50)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: {model.__class__.__name__}")
print(
    f"Optimizer: AdamW (lr={optim.param_groups[0]['lr']}, weight_decay={optim.param_groups[0]['weight_decay']})"
)
print(f"Loss Function: {loss_function.__class__.__name__}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Gradient Accumulation Steps: {accumulation_steps}")
print(f"Effective Batch Size: {batch_size * accumulation_steps}")
print(f"Max Gradient Norm: {max_grad_norm}")
print(f"Device: {device}")
print(f"Number of Training Files: {len(h5_files)}")
print("\nInitial Memory Status:")
get_ram_usage()
print("=" * 50 + "\n")

# Training loop
total_start_time = time()
for epoch in range(epochs):
    epoch_start_time = time()
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)

    # Shuffle h5 files for each epoch
    np.random.shuffle(h5_files)

    model.train()
    epoch_loss = 0
    batch_count = 0
    total_val_loss = 0
    total_val_batches = 0
    total_val_accuracy = 0

    # Process each file
    for file_idx, (
        h5_file,
        (train_frames, train_inputs, val_frames, val_inputs),
    ) in enumerate(batch_generator(h5_files)):
        file_start_time = time()
        n_samples = len(train_frames) + len(val_frames)

        print(f"\nFile {file_idx + 1}/{len(h5_files)}: {os.path.basename(h5_file)}")
        print(f"Total samples: {n_samples}")
        print(f"Training samples: {len(train_frames)}")
        print(f"Validation samples: {len(val_frames)}")

        # Training phase
        train_dataset = TensorDataset(
            torch.Tensor(train_frames), torch.Tensor(train_inputs)
        )
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        file_batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_start_time = time()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Mixed precision training
            with autocast("cuda"):
                outputs = model(batch_X)
                loss = loss_function(outputs, batch_y)
                loss = loss / accumulation_steps

                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_count}. Skipping batch.")
                    optim.zero_grad()
                    continue

                scaler.scale(loss).backward()

                if (batch_count + 1) % accumulation_steps == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1
            file_batch_count += 1

            # Batch logging
            if batch_count % 10 == 0:
                batch_time = time() - batch_start_time
                batches_remaining = len(train_loader) - file_batch_count
                eta = batch_time * batches_remaining

                print(
                    f"Batch: {file_batch_count}/{len(train_loader)} "
                    f"({format_percentage(file_batch_count, len(train_loader))}) | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time/batch: {batch_time:.2f}s | "
                    f"File ETA: {format_time(eta)}"
                )

            # Memory cleanup
            del batch_X, batch_y, outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Validation phase
        model.eval()
        val_start_time = time()
        with torch.no_grad():
            val_dataset = TensorDataset(
                torch.Tensor(val_frames), torch.Tensor(val_inputs)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size * 2, pin_memory=True
            )

            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)

                with autocast("cuda"):
                    val_outputs = model(val_X)
                    batch_val_loss = loss_function(val_outputs, val_y)

                    # Calculate accuracy
                    val_preds = torch.sigmoid(val_outputs) >= 0.5
                    batch_accuracy = (val_preds == val_y).float().mean().item()

                total_val_loss += batch_val_loss.item()
                total_val_batches += 1
                total_val_accuracy = (
                    batch_accuracy
                    if total_val_batches == 1
                    else total_val_accuracy
                    * (total_val_batches - 1)
                    / total_val_batches
                    + batch_accuracy / total_val_batches
                )

                del val_X, val_y, val_outputs, val_preds
                torch.cuda.empty_cache()

        # File summary
        file_time = time() - file_start_time
        val_time = time() - val_start_time
        file_val_loss = (
            total_val_loss / total_val_batches
            if total_val_batches > 0
            else float("inf")
        )

        print(f"\nFile Summary:")
        print(f"Validation Accuracy: {total_val_accuracy:.2%}")
        print(f"File processing time: {format_time(file_time)}")
        print(f"Training time: {format_time(file_time - val_time)}")
        print(f"Validation time: {format_time(val_time)}")
        print(f"Validation Loss: {file_val_loss:.4f}")

        model.train()

    # Epoch summary
    epoch_time = time() - epoch_start_time
    avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
    avg_val_loss = (
        total_val_loss / total_val_batches if total_val_batches > 0 else float("inf")
    )

    print(f"\nEpoch {epoch + 1} Summary:")
    print("-" * 50)
    print(f"Duration: {format_time(epoch_time)}")
    print(f"Training Loss: {avg_epoch_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {total_val_accuracy:.2%}")
    print(f"Batches Processed: {batch_count}")
    print(f"Average Time per Batch: {epoch_time / batch_count:.2f}s")
    print("\nMemory Status:")
    get_ram_usage()

    # if epoch == 0:
    #     model.freeze_input_reduction()
    #     print("\nInput reduction layer frozen for memory optimization")

# Training summary
total_time = time() - total_start_time
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
print(f"Total Duration: {format_time(total_time)}")
print(f"Average Time per Epoch: {format_time(total_time / epochs)}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Final Memory Status:")
get_ram_usage()
print("=" * 50)

# Save the model
torch.save(model.state_dict(), "rocket_model.pth")
print("\nModel saved successfully")
