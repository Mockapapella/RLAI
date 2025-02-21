import time

import evdev
import numpy as np
from evdev import ecodes


class ControllerReader:
    def __init__(self):
        self.device = self._find_controller()
        self.button_states = {}
        self.axis_states = {}

        if self.device:
            self.device.grab()
            self._init_states()
            print(f"Found controller: {self.device.name}")
        else:
            raise RuntimeError("No controller found!")

    def _find_controller(self):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            if evdev.ecodes.EV_KEY in dev.capabilities():
                return dev
        return None

    def _init_states(self):
        # Initialize all buttons to 0
        for code in self.device.active_keys():
            if code in evdev.ecodes.BTN:
                self.button_states[code] = 0

        # Initialize axes from ABS capabilities
        if evdev.ecodes.EV_ABS in self.device.capabilities():
            for code, absinfo in self.device.capabilities()[evdev.ecodes.EV_ABS]:
                self.axis_states[code] = absinfo.value

    def get_state_vector(self) -> np.ndarray:
        """Returns normalized [buttons..., axes...] vector (19 elements)"""
        # Process any pending events
        try:
            for event in self.device.read():
                if event.type == ecodes.EV_KEY:
                    self.button_states[event.code] = event.value
                elif event.type == ecodes.EV_ABS:
                    self.axis_states[event.code] = event.value
        except OSError:
            # Handle potential device disconnection
            time.sleep(0.1)

        # Use explicit button/axis codes matching SimpleController
        BUTTON_CODES = [
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

        AXIS_CODES = [
            ecodes.ABS_X,
            ecodes.ABS_Y,
            ecodes.ABS_RX,
            ecodes.ABS_RY,
            ecodes.ABS_Z,
            ecodes.ABS_RZ,
            ecodes.ABS_HAT0X,
            ecodes.ABS_HAT0Y,
        ]

        buttons = np.array(
            [self.button_states.get(code, 0) for code in BUTTON_CODES], dtype=np.float32
        )

        axes = np.array(
            [self._normalize_axis(code, self.axis_states.get(code, 0)) for code in AXIS_CODES],
            dtype=np.float32,
        )

        return np.concatenate([buttons, axes])

    def _normalize_axis(self, code, value):
        absinfo = self.device.absinfo(code)
        return (value - absinfo.min) / (absinfo.max - absinfo.min)

    def get_active_buttons(self):
        """Returns a list of currently pressed button names"""
        button_names = {
            ecodes.BTN_SOUTH: "BTN_SOUTH",
            ecodes.BTN_EAST: "BTN_EAST",
            ecodes.BTN_NORTH: "BTN_NORTH",
            ecodes.BTN_WEST: "BTN_WEST",
            ecodes.BTN_TL: "BUMPER_RIGHT",
            ecodes.BTN_TR: "BUMPER_LEFT",
            ecodes.BTN_SELECT: "Select",
            ecodes.BTN_START: "Start",
            ecodes.BTN_MODE: "Mode",
            ecodes.BTN_THUMBL: "L3",
            ecodes.BTN_THUMBR: "R3",
        }

        return [
            button_names[code]
            for code, state in self.button_states.items()
            if state == 1 and code in button_names
        ]

    def get_axis_states(self):
        """Returns a dictionary of current axis values"""
        axis_names = {
            ecodes.ABS_X: "Left X",
            ecodes.ABS_Y: "Left Y",
            ecodes.ABS_RX: "Right X",
            ecodes.ABS_RY: "Right Y",
            ecodes.ABS_Z: "LT",
            ecodes.ABS_RZ: "RT",
            ecodes.ABS_HAT0X: "D-Pad X",
            ecodes.ABS_HAT0Y: "D-Pad Y",
        }

        return {
            axis_names[code]: self._normalize_axis(code, value)
            for code, value in self.axis_states.items()
            if code in axis_names
        }


def main():
    try:
        controller = ControllerReader()
        print("Press Ctrl+C to exit")

        while True:
            # Update controller state
            controller.get_state_vector()

            # Clear the line and print current state
            print("\r" + " " * 100, end="", flush=True)  # Clear line
            # Just print the raw state vector values, flushed
            state_vector = controller.get_state_vector()
            print(
                "\r" + " " * 100 + "\r" + " ".join(f"{x:.2f}" for x in state_vector),
                end="",
                flush=True,
            )

            print("", flush=True)
            time.sleep(0.05)  # Small delay to prevent CPU overuse

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
