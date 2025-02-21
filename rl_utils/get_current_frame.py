"""Module for capturing frames from the active window in Rocket League."""

import subprocess
import time
from typing import Optional

import mss
import mss.tools
import numpy as np


class FrameGrabber:
    """Captures and processes frames from the active window.

    Uses mss for efficient screen capture and xdotool for window geometry tracking.
    Implements caching and rate limiting to minimize system resource usage.
    """

    def __init__(self):
        self.sct = mss.mss()
        self.monitor = None
        self.last_geometry_check = 0

    def _get_window_geometry(self) -> Optional[dict]:
        """Get the active window's geometry with caching and rate limiting.

        Returns:
            Dictionary containing window geometry (left, top, width, height) or None if unavailable.
        """
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
        """Capture a frame from the active window efficiently.

        Uses zero-copy transfer when possible to minimize memory usage and improve performance.

        Returns:
            Numpy array containing the captured frame in RGB format, or None if capture failed.
        """
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
    except KeyboardInterrupt:
        print("\nCapture stopped.")
