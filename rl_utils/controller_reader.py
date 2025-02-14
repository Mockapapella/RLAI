import evdev
from evdev import ecodes
import numpy as np
import time

class ControllerReader:
    def __init__(self):
        self.device = self._find_controller()
        self.button_states = {}
        self.axis_states = {}

        if self.device:
            self.device.grab()
            self._init_states()

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
            ecodes.BTN_SOUTH, ecodes.BTN_EAST, ecodes.BTN_NORTH, ecodes.BTN_WEST,
            ecodes.BTN_TL, ecodes.BTN_TR, ecodes.BTN_SELECT, ecodes.BTN_START,
            ecodes.BTN_MODE, ecodes.BTN_THUMBL, ecodes.BTN_THUMBR
        ]

        AXIS_CODES = [
            ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX, ecodes.ABS_RY,
            ecodes.ABS_Z, ecodes.ABS_RZ, ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y
        ]

        buttons = np.array([self.button_states.get(code, 0)
                          for code in BUTTON_CODES], dtype=np.float32)

        axes = np.array([self._normalize_axis(code, self.axis_states.get(code, 0))
                       for code in AXIS_CODES], dtype=np.float32)

        return np.concatenate([buttons, axes])

    def _normalize_axis(self, code, value):
        absinfo = self.device.absinfo(code)
        return (value - absinfo.min) / (absinfo.max - absinfo.min)
