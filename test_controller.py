# test_virtual_controller_advanced.py
import evdev
from evdev import ecodes
import time
import numpy as np
from rl_utils.apply_inputs import InputApplier
from colorama import Fore, Style, init
from datetime import datetime
import logging
import textwrap

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    filename="virtual_controller_test.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class VirtualControllerTester:
    def __init__(self):
        self.applier = InputApplier()
        self.reader = self._initialize_controller()
        self.test_results = []
        self.start_time = time.time()

        # Controller specification from apply_inputs.py
        self.expected_buttons = [
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

        self.expected_axes = [
            ecodes.ABS_X,
            ecodes.ABS_Y,
            ecodes.ABS_RX,
            ecodes.ABS_RY,
            ecodes.ABS_Z,
            ecodes.ABS_RZ,
            ecodes.ABS_HAT0X,
            ecodes.ABS_HAT0Y,
        ]

    def _initialize_controller(self):
        """Initialize controller with connection retries and diagnostics"""
        retries = 3
        for attempt in range(retries):
            try:
                time.sleep(0.5 * attempt)
                reader = VirtualControllerReader()
                logging.info(
                    f"Successfully connected to virtual controller on attempt {attempt + 1}"
                )
                return reader
            except RuntimeError as e:
                logging.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    self._print_device_list()
                    raise RuntimeError(
                        "Failed to connect to virtual controller after 3 attempts"
                    )
        return None

    def _print_device_list(self):
        """Show available input devices for troubleshooting"""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        print(f"\n{Fore.RED}Available input devices:{Style.RESET_ALL}")
        for i, dev in enumerate(devices):
            print(f"{i + 1}. {dev.name} ({dev.path})")
            print(
                f"   Capabilities: {textwrap.shorten(str(dev.capabilities()), width=100)}"
            )

    def _log_test_result(self, name, success, details):
        """Record test results with rich metadata"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "test_name": name,
            "success": success,
            "details": details,
            "duration": time.time() - self.start_time,
        }
        self.test_results.append(result)
        logging.info(f"Test '{name}' {'PASSED' if success else 'FAILED'} - {details}")

    def _validate_state(self, actual, expected, tolerance=0.05):
        """Validate controller state against expected values with detailed diff"""
        errors = []

        # Validate buttons
        for code, exp_val in expected["buttons"].items():
            act_val = actual["buttons"].get(code, -1)
            if act_val != exp_val:
                button_name = ecodes.BTN.get(code, f"BTN_{code}")
                if isinstance(button_name, tuple):
                    button_name = button_name[0]
                errors.append(
                    f"Button {button_name}: Expected {exp_val}, Got {act_val}"
                )

        # Validate axes
        for code, exp_val in expected["axes"].items():
            act_val = actual["axes"].get(code, -1)
            if not (abs(act_val - exp_val) <= tolerance):
                axis_name = ecodes.ABS.get(code, f"ABS_{code}")
                if isinstance(axis_name, tuple):
                    axis_name = axis_name[0]
                errors.append(
                    f"Axis {axis_name}: Expected {exp_val:.2f}, Got {act_val:.2f} "
                    f"(Δ {abs(act_val - exp_val):.2f})"
                )

        return errors

    def _print_colored_state(self, state, errors=None):
        """Print controller state with color-coded annotations"""
        print(f"\n{Fore.CYAN}=== CONTROLLER STATE ==={Style.RESET_ALL}")

        # Print buttons
        print(f"{Fore.YELLOW}Buttons:{Style.RESET_ALL}")
        for code in self.expected_buttons:
            value = state["buttons"].get(code, -1)
            status = f"{Fore.GREEN}✔" if value == 1 else f"{Fore.RED}✗"

            # Get button name safely
            button_name = ecodes.BTN.get(code, f"BTN_{code}")
            if isinstance(button_name, tuple):
                button_name = button_name[0]

            has_error = False
            if errors:
                for error in errors:
                    if (
                        isinstance(button_name, str)
                        and f"Button {button_name}" in error
                    ):
                        has_error = True
                        break

            if has_error:
                status = f"{Fore.RED}⚠"

            print(f"  {button_name:<15} {status} {Style.RESET_ALL}")

        # Print axes
        print(f"\n{Fore.YELLOW}Axes:{Style.RESET_ALL}")
        for code in self.expected_axes:
            value = state["axes"].get(code, -1)
            status = ""

            # Get axis name safely
            axis_name = ecodes.ABS.get(code, f"ABS_{code}")
            if isinstance(axis_name, tuple):
                axis_name = axis_name[0]

            has_error = False
            if errors:
                for error in errors:
                    if isinstance(axis_name, str) and f"Axis {axis_name}" in error:
                        has_error = True
                        break

            if has_error:
                status = f"{Fore.RED}⚠"

            print(f"  {axis_name:<15} {value:6.2f} {status}{Style.RESET_ALL}")

    def run_diagnostic_suite(self):
        """Comprehensive test sequence with automatic validation"""
        test_cases = [
            {
                "name": "All Buttons Pressed",
                "input": np.array([1] * 11 + [0] * 8, dtype=np.float32),
                "expected": {
                    "buttons": {code: 1 for code in self.expected_buttons},
                    "axes": {code: 0 for code in self.expected_axes},
                },
            },
            {
                "name": "Left Stick Full Right",
                "input": np.array(
                    [0] * 11 + [1, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32
                ),
                "expected": {
                    "buttons": {code: 0 for code in self.expected_buttons},
                    "axes": {
                        ecodes.ABS_X: 255,
                        ecodes.ABS_Y: 128,
                        **{
                            code: 0
                            for code in self.expected_axes
                            if code not in [ecodes.ABS_X, ecodes.ABS_Y]
                        },
                    },
                },
            },
            {
                "name": "Right Trigger Full Press",
                "input": np.array([0] * 15 + [1, 0, 0], dtype=np.float32),
                "expected": {
                    "buttons": {code: 0 for code in self.expected_buttons},
                    "axes": {
                        ecodes.ABS_RZ: 255,
                        **{
                            code: 0
                            for code in self.expected_axes
                            if code != ecodes.ABS_RZ
                        },
                    },
                },
            },
        ]

        print(f"\n{Fore.BLUE}=== STARTING DIAGNOSTIC SUITE ==={Style.RESET_ALL}")

        for case in test_cases:
            print(f"\n{Fore.MAGENTA}🏁 TEST: {case['name']}{Style.RESET_ALL}")
            try:
                # Apply test input
                self.applier.apply_inputs(case["input"])
                time.sleep(0.2)  # Allow for event propagation

                # Capture current state
                actual_state = self.reader.get_current_state()

                # Validate results
                errors = self._validate_state(actual_state, case["expected"])
                self._print_colored_state(actual_state, errors)

                if errors:
                    print(f"\n{Fore.RED}❌ TEST FAILED:{Style.RESET_ALL}")
                    for error in errors:
                        print(f"  - {error}")
                    self._log_test_result(
                        case["name"], False, f"{len(errors)} validation errors"
                    )
                else:
                    print(f"\n{Fore.GREEN}✅ TEST PASSED{Style.RESET_ALL}")
                    self._log_test_result(case["name"], True, "All validations passed")

            except Exception as e:
                logging.error(f"Test '{case['name']} failed with exception: {str(e)}")
                print(f"{Fore.RED}🔥 CRITICAL ERROR: {str(e)}{Style.RESET_ALL}")

        self._print_summary_report()

    def _print_summary_report(self):
        """Generate final summary report with statistics"""
        print(f"\n{Fore.BLUE}=== TEST SUMMARY ==={Style.RESET_ALL}")
        passed = sum(1 for r in self.test_results if r["success"])
        failed = len(self.test_results) - passed

        print(f"Tests Run:    {len(self.test_results)}")
        print(f"Passed:       {Fore.GREEN if passed else ''}{passed}{Style.RESET_ALL}")
        print(f"Failed:       {Fore.RED if failed else ''}{failed}{Style.RESET_ALL}")
        print(f"Total Time:   {time.time() - self.start_time:.2f}s")

        if failed > 0:
            print(f"\n{Fore.RED}🔍 FAILURE ANALYSIS:{Style.RESET_ALL}")
            for result in self.test_results:
                if not result["success"]:
                    print(f"- {result['test_name']}: {result['details']}")

    def realtime_monitor(self):
        """Real-time input monitoring with visualization"""
        print(f"\n{Fore.BLUE}=== REALTIME MONITOR ==={Style.RESET_ALL}")
        print("Testing Right Trigger (RZ)")
        print("Press Ctrl+C to exit monitoring mode")
        try:
            while True:
                test_input = np.zeros(19, dtype=np.float32)

                # RZ is at index 15 in our input vector
                # Create a smooth oscillation between 0 and 1
                trigger_value = 0.5 + 0.5 * np.sin(time.time())  # Slower oscillation
                test_input[15] = trigger_value  # RZ axis

                self.applier.apply_inputs(test_input)
                state = self.reader.get_current_state()

                print("\033c", end="")  # Clear console
                self._print_colored_state(state)
                print(
                    f"\n{Fore.YELLOW}Right Trigger Value: {trigger_value:.2f}{Style.RESET_ALL}"
                )

                time.sleep(0.05)
        except KeyboardInterrupt:
            # Ensure all inputs are reset to zero when exiting
            self.applier.apply_inputs(np.zeros(19, dtype=np.float32))
            print(f"\n{Fore.BLUE}Exiting monitoring mode...{Style.RESET_ALL}")


class VirtualControllerReader:
    def __init__(self):
        self.device = None
        self.button_states = {}
        self.axis_states = {}

        # Try multiple times to initialize and grab the controller
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt {attempt + 1} to initialize virtual controller")
                self.device = self._find_virtual_controller()

                if self.device:
                    self._init_states()
                    logging.info(
                        f"Successfully connected to virtual controller: {self.device.name}"
                    )
                    # Wait a brief moment to ensure initialization is complete
                    time.sleep(0.5)
                    break
                else:
                    logging.warning(
                        f"Attempt {attempt + 1}: Virtual controller not found"
                    )
                    time.sleep(1)  # Wait before retrying
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to initialize virtual controller after {max_attempts} attempts"
                    )

    def _find_virtual_controller(self):
        """Find and prioritize the virtual controller"""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

        # First, check if there are any existing game controllers
        game_controllers = [
            dev
            for dev in devices
            if (
                "Gamepad" in dev.name
                or "Controller" in dev.name
                or "Joystick" in dev.name
            )
        ]

        # Log existing controllers
        if game_controllers:
            logging.info("Found existing controllers:")
            for ctrl in game_controllers:
                logging.info(f"  - {ctrl.name} (path: {ctrl.path})")
                # Attempt to grab exclusive access
                try:
                    ctrl.grab()
                    logging.info(f"    Grabbed exclusive access to {ctrl.name}")
                except Exception as e:
                    logging.warning(f"    Could not grab {ctrl.name}: {str(e)}")

        # Now look for our virtual controller
        virtual_controller = None
        for dev in devices:
            if dev.name == "Virtual Controller":
                virtual_controller = dev
                try:
                    # Try to grab exclusive access
                    virtual_controller.grab()
                    logging.info("Grabbed exclusive access to Virtual Controller")
                except Exception as e:
                    logging.warning(f"Could not grab Virtual Controller: {str(e)}")
                break

        return virtual_controller

    def _init_states(self):
        # Initialize button states
        for code in self.device.capabilities().get(evdev.ecodes.EV_KEY, []):
            self.button_states[code] = 0

        # Initialize axis states
        if evdev.ecodes.EV_ABS in self.device.capabilities():
            for abs_code, abs_info in self.device.capabilities()[evdev.ecodes.EV_ABS]:
                self.axis_states[abs_code] = abs_info.value

    def get_current_state(self):
        try:
            for event in self.device.read():
                if event.type == evdev.ecodes.EV_KEY:
                    self.button_states[event.code] = event.value
                elif event.type == evdev.ecodes.EV_ABS:
                    self.axis_states[event.code] = event.value
        except BlockingIOError:
            pass  # No events available
        except OSError as e:
            logging.error(f"Device error: {str(e)}")
            raise

        return {"buttons": self.button_states.copy(), "axes": self.axis_states.copy()}


if __name__ == "__main__":
    try:
        tester = VirtualControllerTester()
        tester.run_diagnostic_suite()
        tester.realtime_monitor()
    except Exception as e:
        print(f"{Fore.RED}💥 Critical initialization error: {str(e)}{Style.RESET_ALL}")
        logging.critical(f"Initialization failed: {str(e)}")
