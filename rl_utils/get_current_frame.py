import numpy as np
import mss
import time
import subprocess


class FrameGrabber:
    def __init__(self):
        self.sct = mss.mss()
        self.current_window = None
        self.monitor = None
        self.last_geometry_check = 0
        self.frame_count = 0
        self.last_fps_print = time.time()
        self.buffer = None  # Reusable frame buffer
        self._last_shape = (0, 0)

    def _get_window_geometry(self):
        """Cached geometry check with rate limiting"""
        if time.time() - self.last_geometry_check < 0.5:  # Only check twice per second
            return self.monitor

        try:
            # Single xdotool call for both ID and geometry
            output = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
                timeout=0.1,
            ).decode()

            geom = {}
            for line in output.strip().split("\n"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    geom[key] = int(val)

            new_monitor = {
                "left": geom["X"],
                "top": geom["Y"],
                "width": geom["WIDTH"],
                "height": geom["HEIGHT"],
            }

            if new_monitor != self.monitor:
                self.monitor = new_monitor
                print(f"\nNew window: {self.monitor}")

            self.last_geometry_check = time.time()
            return self.monitor

        except (subprocess.TimeoutExpired, KeyError, subprocess.CalledProcessError):
            return self.monitor

    def capture_frame(self):
        monitor = self._get_window_geometry()
        if not monitor:
            return None

        try:
            # Get screenshot data directly as bytes
            sct = self.sct.grab(monitor)

            # Only reallocate buffer when window size changes
            if (sct.height, sct.width) != self._last_shape:
                self.buffer = np.zeros((sct.height, sct.width, 3), dtype=np.uint8)
                self._last_shape = (sct.height, sct.width)

            # Direct byte copy using numpy
            np.copyto(
                self.buffer,
                np.frombuffer(sct.rgb, dtype=np.uint8).reshape(
                    (sct.height, sct.width, 3)
                ),
            )
            return self.buffer

        except mss.ScreenShotError:
            return None


if __name__ == "__main__":
    grabber = FrameGrabber()
    try:
        while True:
            frame = grabber.capture_frame()
            # Add your processing here
    except KeyboardInterrupt:
        print("\nCapture stopped.")
