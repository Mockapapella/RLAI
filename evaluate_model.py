from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_window import WindowMonitor
import cv2
import torch
from rl_utils.models import RocketNet

BATCH_SIZE = 5000  # Original batch size
SAVE_DIR = "data/rocket_league/training/"
FPS_UPDATE_INTERVAL = 1.0  # Update FPS display every second for smoother numbers

# Model definition
model = RocketNet()
model.eval()
device = "cuda"
model.to(device)


def main():
    grabber = FrameGrabber()
    window_monitor = WindowMonitor()

    target_size = (480, 270)

    print("Ready to play Rocket League! Focus window to begin...")

    try:
        with torch.no_grad():
            while True:
                if window_monitor.get_active_window() == "Rocket League":
                    frame = grabber.capture_frame()

                    # Skip processing when no frame is captured
                    if frame is None:
                        continue
                    frame = cv2.resize(frame, target_size)
                    frame = torch.Tensor([frame]).to(device)
                    print(frame.shape)
                    output = model(frame)
                    preds = torch.sigmoid(output) >= 0.5
                    print(preds)
    except KeyboardInterrupt:
        print("Stopping inference/evals.")


if __name__ == "__main__":
    main()
