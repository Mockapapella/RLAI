import evdev
from evdev import InputDevice, ecodes, list_devices
import threading


class ControllerState:
    def __init__(self, device_path):
        self.device = InputDevice(device_path)
        self.button_states = {}  # Maps button codes to 0/1
        self.axis_states = {}  # Maps axis codes to current value
        self.axis_info = {}  # Maps axis codes to (min, max)

        # Initialize buttons
        if ecodes.EV_KEY in self.device.capabilities():
            for code in self.device.capabilities()[ecodes.EV_KEY]:
                if 0x100 <= code <= 0x2FF:  # Gamepad buttons
                    self.button_states[code] = 0

        # Initialize axes and their min/max
        if ecodes.EV_ABS in self.device.capabilities():
            for axis_code, absinfo in self.device.capabilities()[ecodes.EV_ABS]:
                self.axis_states[axis_code] = absinfo.value
                self.axis_info[axis_code] = (absinfo.min, absinfo.max)

        # Thread setup
        self._running = True
        self.thread = threading.Thread(target=self._event_loop)
        self.thread.start()

    def _event_loop(self):
        try:
            for event in self.device.read_loop():
                if not self._running:
                    break
                if event.type == ecodes.EV_KEY and event.code in self.button_states:
                    self.button_states[event.code] = 1 if event.value else 0
                elif event.type == ecodes.EV_ABS and event.code in self.axis_states:
                    self.axis_states[event.code] = event.value
        except OSError:
            pass  # Handle device disconnection

    def get_state(self):
        # Get ordered button values (sorted by code)
        button_codes = sorted(self.button_states.keys())
        buttons = [float(self.button_states[code]) for code in button_codes]

        # Get normalized axis values (sorted by code)
        axes = []
        axis_codes = sorted(self.axis_states.keys())
        for code in axis_codes:
            current = self.axis_states[code]
            min_val, max_val = self.axis_info[code]
            normalized = (current - min_val) / (max_val - min_val)
            axes.append(normalized)

        return buttons + axes  # Combine into target vector

    def stop(self):
        self._running = False
        self.thread.join()


def select_device():
    devices = [InputDevice(path) for path in list_devices()]
    print("Available devices:")
    for i, dev in enumerate(devices):
        print(f"{i}: {dev.name} ({dev.path})")
    choice = int(input("Enter device number: "))
    return devices[choice].path


if __name__ == "__main__":
    device_path = select_device()
    controller = ControllerState(device_path)
    try:
        while True:
            input("Press Enter to capture state (Ctrl+C to exit)...")
            state = controller.get_state()
            print("Current state:", state)
    except KeyboardInterrupt:
        controller.stop()
