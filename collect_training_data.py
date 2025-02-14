from rl_utils.get_current_frame import FrameGrabber
from rl_utils.controller_reader import ControllerReader
from rl_utils.data_batcher import BatchSaver
from rl_utils.get_current_window import get_active_window
import time

BATCH_SIZE = 750
SAVE_DIR = "training_data/"
FPS_UPDATE_INTERVAL = 0.1  # Update FPS display every second


def main():
    grabber = FrameGrabber()
    controller = ControllerReader()
    batcher = BatchSaver(BATCH_SIZE, SAVE_DIR)

    print("Starting capture - focus target window to begin...")

    frame_count = 0
    last_fps_time = time.time()
    last_fps = 0

    try:
        while True:
            if get_active_window() == "Alacritty":
                frame = grabber.capture_frame()
                inputs = controller.get_state_vector()
                batcher.add_sample(frame, inputs)

                # Update FPS counter
                frame_count += 1
                current_time = time.time()
                time_diff = current_time - last_fps_time

                if time_diff >= FPS_UPDATE_INTERVAL:
                    last_fps = frame_count / time_diff
                    print(f"\rFPS: {last_fps:.1f}", end="", flush=True)
                    frame_count = 0
                    last_fps_time = current_time

    except KeyboardInterrupt:
        print("\nFinalizing...")
        if batcher.frames:  # Save remaining samples
            batcher._save_batch()
        print("Capture complete!")


if __name__ == "__main__":
    main()
