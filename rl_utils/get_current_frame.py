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
        """Ultra-fast capture path optimized for numpy"""
        monitor = self._get_window_geometry()
        if not monitor:
            return None

        try:
            # Direct capture to numpy array with minimal conversions
            frame = np.array(self.sct.grab(monitor), dtype=np.uint8, order="C")[
                :, :, :3
            ]  # Remove alpha channel if present
            return frame

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
