from evdev.uinput import UInput
from evdev.device import AbsInfo
from evdev import ecodes
import torch
import numpy as np
import logging
from datetime import datetime


class InputApplier:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        if debug_mode:
            logging.basicConfig(
                filename=f"rocket_league_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

        # Define capabilities matching the test controller
        self.capabilities = {
            ecodes.EV_KEY: [
                ecodes.BTN_SOUTH,  # A
                ecodes.BTN_EAST,  # B
                ecodes.BTN_NORTH,  # Y
                ecodes.BTN_WEST,  # X
                ecodes.BTN_TL,  # LB
                ecodes.BTN_TR,  # RB
                ecodes.BTN_SELECT,  # Back
                ecodes.BTN_START,  # Start
                ecodes.BTN_MODE,  # Guide
                ecodes.BTN_THUMBL,  # Left Stick Press
                ecodes.BTN_THUMBR,  # Right Stick Press
            ],
            ecodes.EV_ABS: [
                (
                    ecodes.ABS_X,
                    AbsInfo(value=128, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Left Stick X
                (
                    ecodes.ABS_Y,
                    AbsInfo(value=128, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Left Stick Y
                (
                    ecodes.ABS_RX,
                    AbsInfo(value=128, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Right Stick X
                (
                    ecodes.ABS_RY,
                    AbsInfo(value=128, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Right Stick Y
                (
                    ecodes.ABS_Z,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Left Trigger
                (
                    ecodes.ABS_RZ,
                    AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
                ),  # Right Trigger
                (
                    ecodes.ABS_HAT0X,
                    AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
                ),  # D-pad X
                (
                    ecodes.ABS_HAT0Y,
                    AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
                ),  # D-pad Y
            ],
        }

        try:
            self.ui = UInput(self.capabilities, name="Virtual Controller")
            if self.debug_mode:
                logging.info("Virtual controller initialized successfully")
            print("Virtual controller created successfully")
        except Exception as e:
            error_msg = f"Failed to create virtual controller: {str(e)}"
            if self.debug_mode:
                logging.error(error_msg)
            raise RuntimeError(error_msg)

    def apply_inputs(self, predictions):
        """
        Apply the neural network predictions to the virtual controller

        Args:
            predictions: torch tensor of shape (1, 19) or numpy array of shape (19,)
                       Values should be in range [0, 1] for normalization
        """
        try:
            # Convert predictions to numpy if needed
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()

            if predictions.ndim == 2:
                predictions = predictions.squeeze(0)

            if self.debug_mode:
                logging.debug(f"Raw predictions: {predictions}")

            # Process button inputs (first 11 values)
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

            # Apply button thresholding with hysteresis
            for code, value in zip(button_codes, predictions[:11]):
                button_state = int(value > 0.7)  # Simple threshold for buttons
                if code == ecodes.BTN_MODE or code == ecodes.BTN_START:
                    continue
                self.ui.write(ecodes.EV_KEY, code, button_state)
                if self.debug_mode and button_state:
                    logging.debug(f"Button {code} pressed with value {value:.3f}")

            # Process analog inputs (next 6 values for sticks/triggers)
            stick_trigger_values = predictions[11:17]

            # Process stick and trigger values separately
            scaled_values = np.zeros_like(stick_trigger_values)

            # Process joystick axes (first 4 values)
            for i in [0, 1, 2, 3]:  # Stick axes
                # Center around 0.5 (neutral)
                centered_value = stick_trigger_values[i] - 0.5

                deadzone = 0.025  # 5% deadzone for all stick movement
                sensitivity = 1.0  # Base sensitivity

                # Apply deadzone
                if abs(centered_value) < deadzone:
                    scaled_values[i] = 128  # Center position
                else:
                    # Apply base scaling first
                    adjusted_value = centered_value * sensitivity
                    base_scaled = (adjusted_value + 0.5) * 255

                    # Multiply/divide final values based on stick
                    if i in [0, 1]:  # Left stick (movement)
                        scaled_values[i] = np.clip(
                            base_scaled * 2, 0, 255
                        )  # Double the value
                    else:  # Right stick (camera)
                        scaled_values[i] = np.clip(
                            base_scaled / 2, 0, 255
                        )  # Half the value

            # Process triggers (last 2 values)
            scaled_values[4:6] = np.clip(stick_trigger_values[4:6] * 255, 0, 255)

            # Disable/Override dpad values
            dpad_values = np.array([0, 0]).astype(int)

            # Combine all analog values
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

            analog_values = np.concatenate([scaled_values, dpad_values])

            # Apply analog values
            for code, value in zip(analog_codes, analog_values):
                self.ui.write(ecodes.EV_ABS, code, int(value))
                if self.debug_mode:
                    logging.debug(f"Analog {code} set to {value:.3f}")

            # Print final commands
            print("\nFinal controller commands:")
            print("Button States:")
            for code, value in zip(button_codes, predictions[:11]):
                if code not in [ecodes.BTN_NORTH, ecodes.BTN_MODE, ecodes.BTN_START]:
                    button_state = int(value > 0.7)
                    print(
                        f"  {ecodes.BTN[code]}: {'Pressed' if button_state else 'Released'}"
                    )

            print("\nAnalog Values:")
            analog_names = [
                "Left Stick X",
                "Left Stick Y",
                "Right Stick X",
                "Right Stick Y",
                "Left Trigger",
                "Right Trigger",
                "D-Pad X",
                "D-Pad Y",
            ]
            for name, code, value in zip(analog_names, analog_codes, analog_values):
                print(f"  {name}: {int(value)}")

            # Sync all events
            self.ui.syn()

        except Exception as e:
            error_msg = f"Error applying inputs: {str(e)}"
            if self.debug_mode:
                logging.error(error_msg)
            raise RuntimeError(error_msg)

    def close(self):
        """Safely close the virtual controller"""
        try:
            # Reset all inputs to neutral
            self.apply_inputs(np.array([0] * 11 + [0.5] * 6 + [0] * 2))
            self.ui.close()
            if self.debug_mode:
                logging.info("Virtual controller closed successfully")
        except Exception as e:
            if self.debug_mode:
                logging.error(f"Error closing controller: {str(e)}")


def test_controller():
    """Test function to verify controller functionality"""
    applier = EnhancedInputApplier(debug_mode=True)
    try:
        # Test neutral position
        print("\nTesting neutral position...")
        test_input = np.zeros(19)
        test_input[11:17] = [
            0.5,
            0.5,
            0.5,
            0.5,
            0,
            0,
        ]  # Center sticks, triggers released
        applier.apply_inputs(test_input)

        # Test full left
        print("\nTesting full left...")
        test_input = np.zeros(19)
        test_input[11:17] = [0.0, 0.5, 0.5, 0.5, 0, 0]  # Left stick full left
        applier.apply_inputs(test_input)

        # Test full right
        print("\nTesting full right...")
        test_input = np.zeros(19)
        test_input[11:17] = [1.0, 0.5, 0.5, 0.5, 0, 0]  # Left stick full right
        applier.apply_inputs(test_input)

        # Test partial movement
        print("\nTesting partial right...")
        test_input = np.zeros(19)
        test_input[11:17] = [0.75, 0.5, 0.5, 0.5, 0, 0]  # Left stick 75% right
        applier.apply_inputs(test_input)

    except Exception as e:
        print(f"Test failed: {str(e)}")
    finally:
        applier.close()


if __name__ == "__main__":
    test_controller()
