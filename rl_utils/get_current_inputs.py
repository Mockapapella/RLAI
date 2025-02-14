import evdev
from evdev import InputDevice, UInput, ecodes, list_devices
import threading
import time
from typing import List, Optional, Dict


class SimpleController:
    """Minimal interface for controller input/output as normalized vectors"""

    # Fixed vector layout
    BUTTON_CODES = [
        ecodes.BTN_SOUTH,  # 0: A/Cross
        ecodes.BTN_EAST,  # 1: B/Circle
        ecodes.BTN_NORTH,  # 2: Y/Triangle
        ecodes.BTN_WEST,  # 3: X/Square
        ecodes.BTN_TL,  # 4: Left Bumper
        ecodes.BTN_TR,  # 5: Right Bumper
        ecodes.BTN_SELECT,  # 6: Select/Back
        ecodes.BTN_START,  # 7: Start
        ecodes.BTN_MODE,  # 8: Guide/Home
        ecodes.BTN_THUMBL,  # 9: Left Stick Click
        ecodes.BTN_THUMBR,  # 10: Right Stick Click
    ]

    AXIS_CODES = [
        ecodes.ABS_X,  # 11: Left Stick X
        ecodes.ABS_Y,  # 12: Left Stick Y
        ecodes.ABS_RX,  # 13: Right Stick X
        ecodes.ABS_RY,  # 14: Right Stick Y
        ecodes.ABS_Z,  # 15: Left Trigger
        ecodes.ABS_RZ,  # 16: Right Trigger
        ecodes.ABS_HAT0X,  # 17: D-pad X
        ecodes.ABS_HAT0Y,  # 18: D-pad Y
    ]

    # Standard axis ranges for common controllers
    AXIS_RANGES = {
        ecodes.ABS_X: (0, 255),
        ecodes.ABS_Y: (0, 255),
        ecodes.ABS_RX: (0, 255),
        ecodes.ABS_RY: (0, 255),
        ecodes.ABS_Z: (0, 255),  # Left Trigger
        ecodes.ABS_RZ: (0, 255),  # Right Trigger
        ecodes.ABS_HAT0X: (-1, 1),
        ecodes.ABS_HAT0Y: (-1, 1),
    }

    def __init__(self, create_virtual: bool = True):
        # Initialize device first
        self.device = self._find_gamepad()

        # Initialize state with actual values
        self.button_states = {}
        self.axis_states = {}
        self.axis_info = {}

        if self.device:
            # Get initial button states
            for code in self.BUTTON_CODES:
                self.button_states[code] = 0
                # Try to read initial state if available
                active_keys = self.device.active_keys()
                if active_keys is not None:
                    self.button_states[code] = 1 if code in active_keys else 0

            # Get initial axis states
            for code in self.AXIS_CODES:
                absinfo = self.device.absinfo(code)
                if absinfo is not None:
                    self.axis_states[code] = absinfo.value
                else:
                    self.axis_states[code] = 0

        # Find a suitable input device
        self.device = self._find_gamepad()
        self._running = False
        self.monitor_thread = None
        self.virtual_controller = None

        if self.device:
            self._init_axis_info()
            if create_virtual:
                self._create_virtual_controller()

    def _find_gamepad(self) -> Optional[InputDevice]:
        """Find the first suitable gamepad device"""
        for path in list_devices():
            try:
                dev = InputDevice(path)
                capabilities = dev.capabilities()

                # Check if it has at least some gamepad buttons and axes
                has_buttons = any(
                    code in capabilities.get(ecodes.EV_KEY, [])
                    for code in self.BUTTON_CODES
                )
                has_axes = any(
                    code in [c for c, _ in capabilities.get(ecodes.EV_ABS, [])]
                    for code in self.AXIS_CODES[:4]
                )  # At least main sticks

                if has_buttons and has_axes:
                    print(f"Using gamepad: {dev.name}")
                    return dev
            except Exception:
                continue

        print("No suitable gamepad found!")
        return None

    def _init_axis_info(self):
        """Initialize axis info from device or use defaults"""
        if not self.device:
            self.axis_info = self.AXIS_RANGES
            return

        capabilities = self.device.capabilities()
        if ecodes.EV_ABS in capabilities:
            for code, absinfo in capabilities[ecodes.EV_ABS]:
                if code in self.axis_states:
                    self.axis_info[code] = (absinfo.min, absinfo.max)

        # Fill in defaults for any missing axes
        for code in self.axis_states:
            if code not in self.axis_info:
                self.axis_info[code] = self.AXIS_RANGES.get(code, (0, 255))

    def _create_virtual_controller(self):
        """Create a virtual controller (if possible)"""
        if not self.device:
            return

        try:
            # Start with template capabilities
            caps = {ecodes.EV_KEY: self.BUTTON_CODES, ecodes.EV_ABS: []}

            # Add axis capabilities
            for code in self.AXIS_CODES:
                min_val, max_val = self.axis_info[code]
                absinfo = evdev.AbsInfo(
                    value=0, min=min_val, max=max_val, fuzz=0, flat=0, resolution=0
                )
                caps[ecodes.EV_ABS].append((code, absinfo))

            self.virtual_controller = UInput(
                events=caps, name="SimpleVirtualController", version=0x3
            )
            print("Virtual controller created")
        except Exception as e:
            print(f"Failed to create virtual controller: {str(e)}")

    def _monitor_device(self):
        """Monitor the gamepad device for events"""
        if not self.device:
            return

        try:
            self.device.grab()
            for event in self.device.read_loop():
                if not self._running:
                    break

                if event.type == ecodes.EV_KEY and event.code in self.button_states:
                    self.button_states[event.code] = 1 if event.value else 0

                elif event.type == ecodes.EV_ABS and event.code in self.axis_states:
                    self.axis_states[event.code] = event.value

        except Exception as e:
            print(f"Error monitoring device: {str(e)}")
        finally:
            try:
                self.device.ungrab()
            except:
                pass

    def start(self):
        """Start monitoring input"""
        if self._running or not self.device:
            return

        self._running = True
        self.monitor_thread = threading.Thread(target=self._monitor_device)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Monitoring started")

    def stop(self):
        """Stop monitoring and clean up"""
        self._running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None

        if self.virtual_controller:
            self.virtual_controller.close()
            self.virtual_controller = None

        print("Monitoring stopped")

    def get_input_vector(self) -> List[float]:
        """Get the current input state as a normalized vector

        Returns:
            List of 19 floats representing controller state:
            - 13 buttons (0.0 or 1.0): A/B/X/Y, bumpers, triggers, etc.
            - 6 axes (0.0 to 1.0): left_stick(x,y), right_stick(x,y), dpad(x,y)
        """
        vector = []

        # Add button values (0.0 or 1.0)
        for code in self.BUTTON_CODES:
            vector.append(float(self.button_states.get(code, 0)))

        # Add normalized axis values
        for code in self.AXIS_CODES:
            current = self.axis_states.get(code, 0)
            min_val, max_val = self.axis_info.get(code, (0, 255))

            if code in [ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y]:
                # Convert d-pad (-1/0/1) to normalized (0.0/0.5/1.0)
                normalized = (current - min_val) / (max_val - min_val)
            else:
                normalized = (current - min_val) / (max_val - min_val)

            vector.append(max(0.0, min(1.0, normalized)))  # Clamp to 0-1 range

        return vector

    def set_input_vector(self, vector: List[float], hold_time: float = 0.1):
        """Apply an input vector to the virtual controller

        Args:
            vector: List of 19 floats matching the format from get_input_vector()
            hold_time: How long to hold button presses (seconds)
        """
        if not self.virtual_controller:
            print("Virtual controller not available")
            return

        if len(vector) != len(self.BUTTON_CODES) + len(self.AXIS_CODES):
            raise ValueError(
                f"Expected vector of length {len(self.BUTTON_CODES) + len(self.AXIS_CODES)}"
            )

        # Apply button states
        pressed_buttons = []
        for i, code in enumerate(self.BUTTON_CODES):
            if vector[i] > 0.5:
                self.virtual_controller.write(ecodes.EV_KEY, code, 1)
                pressed_buttons.append(code)

        # Apply axis states
        for i, code in enumerate(self.AXIS_CODES, start=len(self.BUTTON_CODES)):
            min_val, max_val = self.axis_info.get(code, (0, 255))
            raw_value = int(min_val + (vector[i] * (max_val - min_val)))
            self.virtual_controller.write(ecodes.EV_ABS, code, raw_value)

        # Sync and wait
        self.virtual_controller.syn()
        if hold_time > 0 and pressed_buttons:
            time.sleep(hold_time)

        # Release buttons
        for code in pressed_buttons:
            self.virtual_controller.write(ecodes.EV_KEY, code, 0)
        self.virtual_controller.syn()

    def print_vector_layout(self):
        """Print a description of the input vector format"""
        print("\nInput Vector Layout (19 elements):")
        print("Buttons (indices 0-12, values 0.0 or 1.0):")
        for i, code in enumerate(self.BUTTON_CODES):
            name = evdev.ecodes.KEY.get(code, f"Unknown ({code})")
            print(f"  [{i}]: {name}")

        print("\nAxes (indices 13-18, values 0.0 to 1.0):")
        for i, code in enumerate(self.AXIS_CODES, start=len(self.BUTTON_CODES)):
            name = evdev.ecodes.ABS.get(code, f"Unknown ({code})")
            print(f"  [{i}]: {name}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Example usage
if __name__ == "__main__":
    controller = SimpleController()
    controller.print_vector_layout()
    try:
        controller.start()
        print("\nMonitoring controller. Press Ctrl+C to exit...")
        while True:
            vector = controller.get_input_vector()
            buttons = vector[:13]
            axes = vector[13:]
            print(f"\rButtons: {buttons} | Axes: {axes}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.stop()
