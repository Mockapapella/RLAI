from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_inputs import InputMonitor
from rl_utils.get_current_window import get_active_window

frame_grabber = FrameGrabber()
input_monitor = InputMonitor()

while True:
    if get_active_window() == "Rocket League":
        frame = frame_grabber.capture_frame()
        inputs = input_monitor.get_inputs()
        print(inputs)
