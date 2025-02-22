"""Module for collecting training data from Rocket League gameplay.

This script captures frames and controller inputs while playing Rocket League,
saving them in batches for later use in training the AI model.
"""

import time

from rl_utils.controller_reader import ControllerReader
from rl_utils.data_batcher import BatchSaver
from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_window import WindowMonitor

BATCH_SIZE = 5000
SAVE_DIR = "rlai-multi-map/"
FPS_UPDATE_INTERVAL = 1.0


def main() -> None:
    """Run the data collection process.

    Continuously captures frames and controller inputs from Rocket League,
    saving them in batches. The process runs until interrupted with Ctrl+C.
    Displays real-time FPS statistics during capture.
    """
    grabber = FrameGrabber()
    controller = ControllerReader()
    batcher = BatchSaver(BATCH_SIZE, SAVE_DIR)
    window_monitor = WindowMonitor()

    print("Starting capture - focus target window to begin...")

    frame_count: float = 0.0
    last_fps_time = time.time()
    last_fps: float = 0.0

    try:
        while True:
            if window_monitor.get_active_window() == "Rocket League":
                frame = grabber.capture_frame()

                # Skip processing when no frame is captured
                if frame is None:
                    continue

                inputs = controller.get_state_vector()
                batcher.add_sample(frame, inputs)

                # Update FPS counter
                frame_count += 1
                current_time = time.time()
                time_diff = current_time - last_fps_time

                if time_diff >= FPS_UPDATE_INTERVAL:
                    last_fps = frame_count / time_diff
                    print(f"\rFPS: {last_fps:.1f}", end="", flush=True)
                    frame_count = 0.0
                    last_fps_time = current_time

    except KeyboardInterrupt:
        print("\nFinalizing...")
        if batcher.frames:  # Save remaining samples
            batcher._save_batch()
        print("Capture complete!")


if __name__ == "__main__":
    main()
