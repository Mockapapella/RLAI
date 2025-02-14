from rl_utils.get_current_frame import FrameGrabber
from rl_utils.get_current_inputs import SimpleController
from rl_utils.get_current_window import get_active_window

frame_grabber = FrameGrabber()
print(get_active_window())

with SimpleController() as controller:
    while True:
        if get_active_window() == "Alacritty":
            frame = frame_grabber.capture_frame()
            inputs = controller.get_input_vector()
            print(inputs)
            # print(inputs)
