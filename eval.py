import logging
from datetime import datetime

import cv2
import torch

from rl_utils.apply_inputs import InputApplier
from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_window import WindowMonitor
from rl_utils.model import RocketNet

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FPS_UPDATE_INTERVAL = 1.0

# Model setup
model = RocketNet()
model.load_state_dict(torch.load("rlai-1.4M/rlai-1.4M.pth"))
model.eval()
device = "cuda"
model.to(device)


def main():
    grabber = FrameGrabber()
    window_monitor = WindowMonitor()
    input_applier = InputApplier(debug_mode=True)

    target_size = (480, 270)

    print("Ready to play Rocket League! Focus window to begin...")
    logging.info("Evaluation script started")

    try:
        with torch.no_grad():
            while True:
                if window_monitor.get_active_window() == "Rocket League":
                    frame = grabber.capture_frame()

                    if frame is None:
                        logging.warning("Failed to capture frame")
                        continue

                    frame = cv2.resize(frame, target_size)
                    frame = torch.Tensor([frame]).to(device)

                    try:
                        output = model(frame)
                        input_applier.apply_inputs(output)

                    except Exception as e:
                        logging.error(f"Error during inference/input application: {str(e)}")
                        continue

    except KeyboardInterrupt:
        print("\nStopping inference/evals.")
        logging.info("Evaluation stopped by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        logging.critical(f"Critical error occurred: {str(e)}")
    finally:
        input_applier.close()
        logging.info("Cleanup completed")


if __name__ == "__main__":
    main()
