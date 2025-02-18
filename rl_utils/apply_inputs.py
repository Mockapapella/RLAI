from evdev.uinput import UInput
from evdev.device import AbsInfo
from evdev import ecodes
import torch
import numpy as np


class InputApplier:
    def __init__(self):
        # Define capabilities for virtual controller
        self.capabilities = {
            ecodes.EV_KEY: [
                ecodes.BTN_SOUTH,  # BTN_SOUTH
                ecodes.BTN_EAST,  # BTN_EAST
                ecodes.BTN_NORTH,  # BTN_NORTH
                ecodes.BTN_WEST,  # BTN_WEST
                ecodes.BTN_TL,  # BTN_TL
                ecodes.BTN_TR,  # BTN_TR
                ecodes.BTN_SELECT,  # BTN_SELECT
                ecodes.BTN_START,  # BTN_START
                ecodes.BTN_MODE,  # BTN_MODE
                ecodes.BTN_THUMBL,  # BTN_THUMBL
                ecodes.BTN_THUMBR,  # BTN_THUMBR
            ],
            ecodes.EV_ABS: [
                (
                    ecodes.ABS_X,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_X
                (
                    ecodes.ABS_Y,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_Y
                (
                    ecodes.ABS_RX,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_RX
                (
                    ecodes.ABS_RY,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_RY
                (
                    ecodes.ABS_Z,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_Z
                (
                    ecodes.ABS_RZ,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # ABS_RZ
                (
                    ecodes.ABS_HAT0X,
                    AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
                ),  # ABS_HAT0X
                (
                    ecodes.ABS_HAT0Y,
                    AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
                ),  # ABS_HAT0Y
            ],
        }

        self.ui = UInput(self.capabilities, name="Virtual Controller")
        print("Virtual controller created")

    def apply_inputs(self, predictions):
        """
        Apply the predicted inputs to the virtual controller
        predictions: torch tensor of shape (1, 19) or numpy array of shape (19,)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if predictions.ndim == 2:
            predictions = predictions.squeeze(0)

        # Apply button inputs (first 11 values)
        button_codes = [
            ecodes.BTN_SOUTH,
            ecodes.BTN_EAST,
            ecodes.BTN_NORTH,
            ecodes.BTN_WEST,
            ecodes.BTN_TL,
            ecodes.BTN_TR,
            ecodes.BTN_SELECT,
            ecodes.BTN_START,
            ecodes.BTN_MODE,
            ecodes.BTN_THUMBL,
            ecodes.BTN_THUMBR,
        ]

        for code, value in zip(button_codes, predictions[:11]):
            self.ui.write(ecodes.EV_KEY, code, int(value > 0.5))

        # Apply analog inputs (next 8 values)
        analog_codes = [
            ecodes.ABS_X,
            ecodes.ABS_Y,
            ecodes.ABS_RX,
            ecodes.ABS_RY,
            ecodes.ABS_Z,
            ecodes.ABS_RZ,
            ecodes.ABS_HAT0X,
            ecodes.ABS_HAT0Y,
        ]

        # Convert normalized values to appropriate ranges
        analog_values = predictions[11:17] * 255  # For joysticks and triggers (0-255)
        dpad_values = (predictions[17:19] * 2 - 1).astype(int)  # For D-pad (-1 to 1)

        # Combine all analog values
        analog_values = np.concatenate([analog_values, dpad_values])

        for code, value in zip(analog_codes, analog_values):
            self.ui.write(ecodes.EV_ABS, code, int(value))

        # Sync the events
        self.ui.syn()

    def close(self):
        """Close the virtual controller"""
        self.ui.close()


def main():
    applier = InputApplier()
    try:
        while True:
            test_input = torch.zeros(1, 19)
            applier.apply_inputs(test_input)
    except KeyboardInterrupt:
        print("\nClosing virtual controller")
        applier.close()


if __name__ == "__main__":
    main()
