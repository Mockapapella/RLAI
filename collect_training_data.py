from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_inputs import SimpleController
from rl_utils.get_current_window import get_active_window
import numpy as np
import os
import time
from threading import Thread

# Pre-allocate buffers for 5000 samples
MAX_BATCH = 2500
FRAME_SHAPE = (1409, 2556, 3)  # Assuming fixed resolution from FrameGrabber
frame_buffer = np.zeros((MAX_BATCH, *FRAME_SHAPE), dtype=np.uint8)
input_buffer = np.zeros((MAX_BATCH, 19), dtype=np.float32)
write_idx = 0
save_thread = None


def save_batch_async(current_frames, current_inputs, batch_dir="training_data"):
    """Save in a background thread to avoid blocking capture"""
    global save_thread

    def save_task(f, i):
        os.makedirs(batch_dir, exist_ok=True)
        ts = time.time_ns()
        np.save(os.path.join(batch_dir, f"{ts}_frames.npy"), f)
        np.save(os.path.join(batch_dir, f"{ts}_inputs.npy"), i)

    if save_thread and save_thread.is_alive():
        save_thread.join()  # Wait for previous save to finish

    save_thread = Thread(
        target=save_task, args=(current_frames.copy(), current_inputs.copy())
    )
    save_thread.start()
    print(f"Started async save of {current_frames.shape[0]} samples")


try:
    with SimpleController() as controller:
        grabber = FrameGrabber()

        # Warm-up capture to initialize frame buffers
        while get_active_window() != "Alacritty":
            time.sleep(0.01)

        # Main capture loop
        while True:
            if get_active_window() == "Alacritty":
                frame = grabber.capture_frame()
                if frame is not None:
                    # Direct buffer writing
                    frame_buffer[write_idx] = frame
                    input_buffer[write_idx] = controller.get_input_vector()
                    write_idx += 1

                    # Save and reset when full
                    if write_idx >= MAX_BATCH:
                        save_batch_async(
                            frame_buffer[:write_idx], input_buffer[:write_idx]
                        )
                        write_idx = 0
            else:
                time.sleep(0.05)  # Longer sleep when inactive

except KeyboardInterrupt:
    print("\nInterrupted. Final save...")
    if write_idx > 0:
        save_batch_async(frame_buffer[:write_idx], input_buffer[:write_idx])
    if save_thread:
        save_thread.join()
    print("Data saved. Exiting.")
