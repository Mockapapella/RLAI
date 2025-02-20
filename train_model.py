import h5py
import glob
import os
import numpy as np
import random
import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from rl_utils.models import RocketNet, compute_improved_loss
from torch.utils.data import TensorDataset, DataLoader
from time import time
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

torch.multiprocessing.set_sharing_strategy("file_system")
writer = SummaryWriter("runs/rocket_league/")


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


def calculate_metrics(outputs, targets):
    """Calculate comprehensive metrics for binary and analog controls"""
    # Split the outputs
    binary_logits = outputs[:, :11]
    analog_values = outputs[:, 11:]

    # Split the targets
    binary_targets = targets[:, :11]
    analog_targets = targets[:, 11:]

    # Binary metrics
    binary_probs = torch.sigmoid(binary_logits)
    binary_preds = (binary_probs >= 0.5).float()
    binary_accuracy = (binary_preds == binary_targets).float().mean().item()

    # Analog metrics with multiple thresholds
    analog_errors = torch.abs(analog_values - analog_targets)
    analog_strict = (analog_errors < 0.01).float().mean().item()  # Within 1%
    analog_usable = (analog_errors < 0.05).float().mean().item()  # Within 5%

    # Use the usable metric for training feedback while tracking strict
    analog_accuracy = analog_usable

    # Calculate combined accuracy
    combined_accuracy = (binary_accuracy + analog_accuracy) / 2

    return binary_accuracy, analog_accuracy, combined_accuracy, analog_strict


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


def process_batch(frames, device):
    """Process and normalize frames"""
    # Convert to torch tensor
    frames_tensor = torch.from_numpy(frames).float() / 255.0
    frames_tensor = frames_tensor.to(device)
    return frames_tensor


def batch_generator(h5_files):
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            frames = f["frames"][:]
            inputs = f["inputs"][:]
        yield h5_file, split_batch(frames, inputs)


# Get files
pattern = os.path.join("data/rocket_league/training/", "*_batch.h5")
h5_files = glob.glob(pattern)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

epochs = 25
batch_size = 128
accumulation_steps = 1  # Gradient accumulation for effective batch size of 512
scaler = GradScaler()  # For mixed precision training
max_grad_norm = 1.0  # Increased from 0.1 for better gradient flow

# Calculate approximate steps
sample_files = min(5, len(h5_files))
approx_steps_per_epoch = 0
for i in range(sample_files):
    with h5py.File(h5_files[i], "r") as f:
        file_samples = min(5000, len(f["frames"]))
        approx_steps_per_epoch += (
            file_samples * 0.8
        ) // batch_size  # 80% for training, batch size 128

# Extrapolate to all files
approx_steps_per_epoch = int(
    approx_steps_per_epoch * (len(h5_files) / max(1, sample_files))
)
total_steps = approx_steps_per_epoch * 25  # 25 epochs max

# Training configuration
model = RocketNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Calculate and display parameter count
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Optimizer with lower learning rate
optim = AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
)

# OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=3e-4,
    total_steps=total_steps,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1000,
)


# Early stopping configuration
early_stopping = {
    "best_val_loss": float("inf"),
    "patience": 5,
    "counter": 0,
    "best_epoch": 0,
}

# DataLoader configuration
dataloader_kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": False,
}

# Log training configuration
print("\n" + "=" * 50)
print("TRAINING CONFIGURATION")
print("=" * 50)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: {model.__class__.__name__}")
print(f"Parameter Count: {param_count:,}")
print(
    f"Optimizer: AdamW (lr={optim.param_groups[0]['lr']}, weight_decay={optim.param_groups[0]['weight_decay']})"
)
print(f"Scheduler: OneCycleLR (max_lr=3e-4, total_steps={total_steps})")
print(f"Loss Function: Improved (Focal + SmoothL1)")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Gradient Accumulation Steps: {accumulation_steps}")
print(f"Effective Batch Size: {batch_size * accumulation_steps}")
print(f"Max Gradient Norm: {max_grad_norm}")
print(f"Device: {device}")
print(f"Number of Training Files: {len(h5_files)}")
print(f"Early Stopping Patience: {early_stopping['patience']}")
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

    # Track metrics
    epoch_metrics = {
        "train_binary_acc": 0,
        "train_analog_acc": 0,
        "train_combined_acc": 0,
        "val_binary_acc": 0,
        "val_analog_acc": 0,
        "val_combined_acc": 0,
    }
    files_processed = 0

    # Process each file
    for file_idx, (
        h5_file,
        (train_frames, train_inputs, val_frames, val_inputs),
    ) in enumerate(batch_generator(h5_files)):
        file_start_time = time()
        n_samples = len(train_frames) + len(val_frames)
        files_processed += 1

        print(f"\nFile {file_idx + 1}/{len(h5_files)}: {os.path.basename(h5_file)}")
        print(f"Total samples: {n_samples}")
        print(f"Training samples: {len(train_frames)}")
        print(f"Validation samples: {len(val_frames)}")

        # Training phase
        train_dataset = TensorDataset(
            torch.from_numpy(train_frames).float(),
            torch.from_numpy(train_inputs).float(),
        )
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        file_batch_count = 0

        # Reset gradients at start of file
        optim.zero_grad()

        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_start_time = time()

            # Process batch
            batch_X = process_batch(batch_X.numpy(), device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)

            # Skip if NaN/Inf outputs detected
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(
                    f"NaN/Inf in outputs detected at batch {batch_count}. Skipping batch."
                )
                continue

            # Calculate loss with improved loss function
            loss = compute_improved_loss(outputs, batch_y)
            loss = loss / accumulation_steps

            # Skip if NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_count}. Skipping batch.")
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Calculate and track metrics
            with torch.no_grad():
                binary_acc, analog_acc, combined_acc, analog_strict = calculate_metrics(
                    outputs, batch_y
                )
                epoch_metrics["train_binary_acc"] += binary_acc
                epoch_metrics["train_analog_acc"] += analog_acc
                epoch_metrics["train_combined_acc"] += combined_acc

            # Step optimizer after accumulation
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                # Unscale before manipulation
                scaler.unscale_(optim)

                # Get average gradient magnitudes for each head
                binary_grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in model.binary_head.parameters() if p.grad is not None])
                )
                analog_grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in model.analog_head.parameters() if p.grad is not None])
                )

                # If binary gradients dominate, scale them down
                if binary_grad_norm > 2.0 * analog_grad_norm:
                    scale_factor = analog_grad_norm / binary_grad_norm
                    for p in model.binary_head.parameters():
                        if p.grad is not None:
                            p.grad *= scale_factor

                # Apply normal gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1
            file_batch_count += 1

            # Log training metrics to tensorboard
            if i % 5 == 0 or i + 1 == len(train_loader):
                global_step = epoch * len(h5_files) * len(train_loader) + batch_count
                writer.add_scalar(
                    "Training/Loss", loss.item() * accumulation_steps, global_step
                )
                writer.add_scalar(
                    "Training/Learning_Rate", scheduler.get_last_lr()[0], global_step
                )
                writer.add_scalar("Training/Binary_Accuracy", binary_acc, global_step)
                writer.add_scalar("Training/Analog_Accuracy", analog_acc, global_step)

                # Log gradients occasionally
                if i % 20 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(
                                f"Gradients/{name}", param.grad, global_step
                            )

            # Print progress
            if i % 5 == 0 or i + 1 == len(train_loader):
                batch_time = time() - batch_start_time
                batches_remaining = len(train_loader) - file_batch_count
                eta = batch_time * batches_remaining

                print(
                    f"Batch: {file_batch_count}/{len(train_loader)} "
                    f"({format_percentage(file_batch_count, len(train_loader))}) | "
                    f"Loss: {loss.item() * accumulation_steps:.4f} | "
                    f"B-Acc: {binary_acc:.2%} | "
                    f"A-Acc: {analog_acc:.2%} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"Time/batch: {batch_time:.2f}s | "
                    f"File ETA: {format_time(eta)}"
                )

        # Validation phase
        model.eval()
        val_start_time = time()
        file_val_metrics = {
            "binary_acc": 0,
            "analog_acc": 0,
            "combined_acc": 0,
            "count": 0,
        }

        with torch.no_grad():
            val_dataset = TensorDataset(
                torch.from_numpy(val_frames).float(),
                torch.from_numpy(val_inputs).float(),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size * 2, pin_memory=True
            )

            for val_X, val_y in val_loader:
                val_X = process_batch(val_X.numpy(), device)
                val_y = val_y.to(device)

                val_outputs = model(val_X)
                batch_val_loss = compute_improved_loss(val_outputs, val_y)

                # Calculate metrics
                binary_acc, analog_acc, combined_acc, analog_strict = calculate_metrics(
                    val_outputs, val_y
                )

                # Accumulate weighted metrics
                samples = len(val_X)
                file_val_metrics["binary_acc"] += binary_acc * samples
                file_val_metrics["analog_acc"] += analog_acc * samples
                file_val_metrics["combined_acc"] += combined_acc * samples
                file_val_metrics["analog_strict"] = analog_strict  # Track strict metric separately
                file_val_metrics["count"] += samples

                total_val_loss += batch_val_loss.item()
                total_val_batches += 1

        # Calculate average validation metrics for the file
        avg_file_metrics = {}
        if file_val_metrics["count"] > 0:
            for key in ["binary_acc", "analog_acc", "combined_acc"]:
                avg_file_metrics[key] = (
                    file_val_metrics[key] / file_val_metrics["count"]
                )
                # Add to epoch total
                epoch_metrics[f"val_{key}"] += avg_file_metrics[key]

        # Calculate average validation loss
        file_val_loss = total_val_loss / max(total_val_batches, 1)

        # Log validation metrics to tensorboard
        file_step = epoch * len(h5_files) + file_idx
        writer.add_scalar("Validation/Loss", file_val_loss, file_step)
        writer.add_scalar(
            "Validation/Binary_Accuracy",
            avg_file_metrics.get("binary_acc", 0),
            file_step,
        )
        writer.add_scalar(
            "Validation/Analog_Accuracy",
            avg_file_metrics.get("analog_acc", 0),
            file_step,
        )
        writer.add_scalar(
            "Validation/Combined_Accuracy",
            avg_file_metrics.get("combined_acc", 0),
            file_step,
        )

        # Log memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # To GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # To GB
        writer.add_scalar("System/GPU_Memory_Allocated_GB", memory_allocated, file_step)
        writer.add_scalar("System/GPU_Memory_Reserved_GB", memory_reserved, file_step)

        # File summary
        file_time = time() - file_start_time
        val_time = time() - val_start_time

        print(f"\nFile Summary:")
        print(f"File processing time: {format_time(file_time)}")
        print(f"Training time: {format_time(file_time - val_time)}")
        print(f"Validation time: {format_time(val_time)}")
        print(f"Validation Loss: {file_val_loss:.4f}")
        print(
            f"Validation Binary Accuracy: {avg_file_metrics.get('binary_acc', 0):.2%}"
        )
        print(
            f"Validation Analog Accuracy: {avg_file_metrics.get('analog_acc', 0):.2%}"
        )

        # Switch back to train mode
        model.train()

    # Calculate final epoch metrics averages
    for key in epoch_metrics:
        epoch_metrics[key] /= files_processed

    # Epoch summary
    epoch_time = time() - epoch_start_time
    avg_epoch_loss = epoch_loss / max(batch_count, 1)
    avg_val_loss = total_val_loss / max(total_val_batches, 1)

    print(f"\nEpoch {epoch + 1} Summary:")
    print("-" * 50)
    print(f"Duration: {format_time(epoch_time)}")
    print(f"Training Loss: {avg_epoch_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Training Binary Accuracy: {epoch_metrics['train_binary_acc']:.2%}")
    print(f"Training Analog Accuracy: {epoch_metrics['train_analog_acc']:.2%}")
    print(f"Validation Binary Accuracy: {epoch_metrics['val_binary_acc']:.2%}")
    print(f"Validation Analog Accuracy: {epoch_metrics['val_analog_acc']:.2%}")
    print(f"Batches Processed: {batch_count}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    print(f"Average Time per Batch: {epoch_time / max(batch_count, 1):.2f}s")
    print("\nMemory Status:")
    get_ram_usage()

    # Save epoch checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if hasattr(scheduler, "state_dict")
            else None,
            "loss": avg_val_loss,
            "metrics": epoch_metrics,
        },
        f"rocket_model_epoch_{epoch + 1}.pth",
    )

    # Early stopping check
    if avg_val_loss < early_stopping["best_val_loss"]:
        early_stopping["best_val_loss"] = avg_val_loss
        early_stopping["counter"] = 0
        early_stopping["best_epoch"] = epoch + 1

        # Save best model separately
        torch.save(model.state_dict(), "rocket_model_best.pth")
        print(f"✓ Saved new best model with validation loss: {avg_val_loss:.4f}")
    else:
        early_stopping["counter"] += 1
        print(
            f"! Validation loss did not improve for {early_stopping['counter']} epochs. "
            f"Best: {early_stopping['best_val_loss']:.4f} at epoch {early_stopping['best_epoch']}"
        )

        if early_stopping["counter"] >= early_stopping["patience"]:
            print(f"⚠ Early stopping triggered after {epoch + 1} epochs")
            break

# Training summary
total_time = time() - total_start_time
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
print(f"Total Duration: {format_time(total_time)}")
print(f"Average Time per Epoch: {format_time(total_time / (epoch + 1))}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(
    f"Best Model: epoch {early_stopping['best_epoch']} with loss {early_stopping['best_val_loss']:.4f}"
)
print("Final Memory Status:")
get_ram_usage()
print("=" * 50)

# Save the final model
torch.save(model.state_dict(), "rocket_model_final.pth")
print("\nFinal model saved successfully")

# Final cleanup
writer.close()
torch.cuda.empty_cache()
