import subprocess
import time
from typing import Optional

import mss
import mss.tools
import numpy as np


class FrameGrabber:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = None
        self.last_geometry_check = 0

    def _get_window_geometry(self) -> Optional[dict]:
        """Cached geometry check with rate limiting"""
        current_time = time.time()
        if current_time - self.last_geometry_check < 0.5:  # Only check twice per second
            return self.monitor

        try:
            # Single xdotool call for both ID and geometry
            output = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
                timeout=0.1,
            ).decode()

            # Fast string parsing without dict comprehension
            geom = {}
            for line in output.strip().split("\n"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    if key in ("X", "Y", "WIDTH", "HEIGHT"):
                        geom[key] = int(val)

            # Only create new dict if values changed
            if not self.monitor or any(
                self.monitor[k] != v
                for k, v in zip(
                    ["left", "top", "width", "height"],
                    [geom["X"], geom["Y"], geom["WIDTH"], geom["HEIGHT"]],
                )
            ):
                self.monitor = {
                    "left": geom["X"],
                    "top": geom["Y"],
                    "width": geom["WIDTH"],
                    "height": geom["HEIGHT"],
                }
                print(f"\nNew window: {self.monitor}")

            self.last_geometry_check = current_time
            return self.monitor

        except (subprocess.TimeoutExpired, KeyError, subprocess.CalledProcessError):
            return self.monitor

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame with zero-copy transfer when possible"""
        monitor = self._get_window_geometry()
        if not monitor:
            return None

        try:
            # Capture and convert frame
            raw = self.sct.grab(monitor)
            frame = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(raw.height, raw.width, 3).copy()
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
