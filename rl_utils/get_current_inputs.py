import evdev
from evdev import ecodes
import time
import sys


class InputMonitor:
    def __init__(self):
        self.current_keys = set()
        self.axis_states = {
            ecodes.ABS_X: 0,  # Left stick X
            ecodes.ABS_Y: 0,  # Left stick Y
            ecodes.ABS_RX: 0,  # Right stick X
            ecodes.ABS_RY: 0,  # Right stick Y
            ecodes.ABS_HAT0X: 0,  # D-pad X
            ecodes.ABS_HAT0Y: 0,  # D-pad Y
        }
        self.devices = self._get_input_devices()

        print("Detected input devices:")
        for i, dev in enumerate(self.devices):
            print(f" {i + 1}. {dev.name} (Path: {dev.path})")

    def _get_input_devices(self):
        devices = []
        for device_path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(device_path)
                # Include all input devices since controllers might not be classified as keyboards/mice
                devices.append(dev)
            except:
                continue
        return devices

    def _get_key_name(self, key_event):
        """Convert keycode tuple to string if needed"""
        if isinstance(key_event.keycode, tuple):
            return "/".join(key_event.keycode)
        return key_event.keycode

    def get_inputs(self):
        try:
            print("\nMonitoring input (Press CTRL+C to exit)...")
            for device in self.devices:
                try:
                    device.grab()
                    print(f"Grabbed device: {device.name}")
                except Exception as e:
                    print(f"Couldn't grab {device.name}: {str(e)}")

            for device in self.devices:
                try:
                    events = device.read()
                    if not events:
                        continue

                    for event in events:
                        if event.type == ecodes.EV_KEY:
                            key_event = evdev.categorize(event)
                            key_name = self._get_key_name(key_event)
                            if key_event.keystate == key_event.key_down:
                                self.current_keys.add(key_name)
                            elif key_event.keystate == key_event.key_up:
                                self.current_keys.discard(key_name)

                        elif event.type == ecodes.EV_ABS:
                            # Handle analog sticks and d-pad
                            if event.code in self.axis_states:
                                self.axis_states[event.code] = event.value

                except BlockingIOError:
                    continue

            # Get formatted axis values
            axis_display = (
                f"L: ({self.axis_states[ecodes.ABS_X]}, {self.axis_states[ecodes.ABS_Y]}) "
                f"R: ({self.axis_states[ecodes.ABS_RX]}, {self.axis_states[ecodes.ABS_RY]}) "
                f"D-pad: ({self.axis_states[ecodes.ABS_HAT0X]}, {self.axis_states[ecodes.ABS_HAT0Y]})"
            )

            output = f"Keys: {sorted(self.current_keys)} | {axis_display}"
            print(f"\r{output.ljust(200)}", end="", flush=True)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            for device in self.devices:
                device.ungrab()
            sys.exit(0)


if __name__ == "__main__":
    monitor = InputMonitor()
    monitor.monitor()
