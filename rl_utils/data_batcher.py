import numpy as np
import time
import os
import cv2
from concurrent.futures import ThreadPoolExecutor


class BatchSaver:
    def __init__(self, batch_size: int, save_dir: str):
        self.batch_size = batch_size
        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.frames = []
        self.inputs = []
        self.target_shape = [640, 480]
        self.executor = ThreadPoolExecutor(max_workers=1)

    def add_sample(self, frame: np.ndarray, inputs: np.ndarray):
        # Set target shape from first frame in batch
        if not self.frames:
            self.target_shape = frame.shape[:2]  # (height, width)
        elif frame.shape[:2] != self.target_shape:
            # Resize frame to match batch dimensions
            frame = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]))

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
        self.target_shape = None  # Reset target shape for next batch

        # Submit save task to thread pool
        def save_task():
            frame_path = os.path.join(self.save_dir, f"{timestamp}_frames.npy")
            input_path = os.path.join(self.save_dir, f"{timestamp}_inputs.npy")

            # Convert to numpy arrays with consistent shapes
            frames_array = np.array(frames_to_save)
            inputs_array = np.array(inputs_to_save)

            np.save(frame_path, frames_array)
            np.save(input_path, inputs_array)

        self.executor.submit(save_task)
