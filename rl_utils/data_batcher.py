import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np


class BatchSaver:
    def __init__(
        self,
        batch_size: int,
        save_dir: str,
        target_size=(480, 270),  # Reduced resolution
        use_grayscale=False,  # Convert to grayscale
        compression_level=1,  # Lower compression for better CPU performance
    ):
        self.batch_size = batch_size
        self.save_dir = os.path.expanduser(save_dir)
        self.target_size = target_size
        self.use_grayscale = use_grayscale
        self.compression_level = compression_level
        os.makedirs(self.save_dir, exist_ok=True)

        self.frames = []
        self.inputs = []
        self.executor = ThreadPoolExecutor(max_workers=1)

    def add_sample(self, frame: np.ndarray, inputs: np.ndarray):
        # Resize frame
        frame = cv2.resize(frame, self.target_size)

        # Convert to grayscale if enabled
        if self.use_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        self.frames.append(frame)
        self.inputs.append(inputs)

        # Save if batch size reached
        if len(self.frames) >= self.batch_size:
            self._save_batch()

    def _save_batch(self):
        if not self.frames:  # Don't save empty batches
            return

        # Create copies of current batch data
        frames_to_save = self.frames.copy()
        inputs_to_save = self.inputs.copy()
        timestamp = int(time.time() * 1000)

        # Clear current batch immediately
        self.frames.clear()
        self.inputs.clear()

        # Submit save task to thread pool
        def save_task():
            # Convert to numpy arrays
            frames_array = np.array(frames_to_save)
            inputs_array = np.array(inputs_to_save)

            # Save to HDF5 format with compression
            filename = os.path.join(self.save_dir, f"{timestamp}_batch.h5")
            with h5py.File(filename, "w") as f:
                # Create compressed datasets
                f.create_dataset(
                    "frames",
                    data=frames_array,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                f.create_dataset(
                    "inputs",
                    data=inputs_array,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )

                # Store metadata
                f.attrs["timestamp"] = timestamp
                f.attrs["original_shape"] = frames_array.shape
                f.attrs["grayscale"] = self.use_grayscale
                f.attrs["target_size"] = self.target_size

        self.executor.submit(save_task)

    def load_batch(self, filename):
        """Utility method to load a saved batch"""
        with h5py.File(filename, "r") as f:
            frames = f["frames"][:]
            inputs = f["inputs"][:]
            metadata = dict(f.attrs)
        return frames, inputs, metadata
