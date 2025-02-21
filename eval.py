from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_window import WindowMonitor
import cv2
import torch
from rl_utils.models import RocketNet
from rl_utils.apply_inputs import EnhancedInputApplier
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f"rocket_league_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BATCH_SIZE = 5000
SAVE_DIR = "data/rocket_league/training/"
FPS_UPDATE_INTERVAL = 1.0

# Model setup
model = RocketNet()
model.load_state_dict(torch.load("rocket_model_best.pth"))
model.eval()
device = "cuda"
model.to(device)
torch.set_printoptions(precision=4, threshold=1000, edgeitems=1000, linewidth=1000)


def main():
    grabber = FrameGrabber()
    window_monitor = WindowMonitor()
    input_applier = EnhancedInputApplier(debug_mode=True)  # Enable debug mode initially

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
                        print(output)
                        # preds = torch.sigmoid(output)

                        # Log predictions for debugging
                        # logging.debug(f"Raw predictions: {preds.cpu().numpy()}")

                        # Apply inputs with error handling
                        input_applier.apply_inputs(output)

                    except Exception as e:
                        logging.error(
                            f"Error during inference/input application: {str(e)}"
                        )
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
