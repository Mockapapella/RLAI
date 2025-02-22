"""Module for monitoring and retrieving the active window title in Rocket League."""

import subprocess
import time


class WindowMonitor:
    """Monitors the active window title with caching and rate limiting.

    Provides efficient window title tracking by implementing caching and
    rate limiting to minimize system calls and resource usage.
    """

    def __init__(self):
        self.last_check = 0
        self.cached_name = None
        self.update_interval = 0.3  # Update every 300ms

    def get_active_window(self) -> str:
        """Get the title of the currently active window.

        Returns:
            The title of the active window, or the last known title if
            the check fails or is rate limited.
        """
        current_time = time.time()
        if current_time - self.last_check < self.update_interval:
            return self.cached_name

        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                check=True,
                timeout=0.05,  # Faster failure if window doesn't respond
            )
            self.cached_name = result.stdout.strip()
        except Exception:
            pass  # Keep last known value

        self.last_check = current_time
        return self.cached_name


if __name__ == "__main__":
    # Example usage of WindowMonitor
    monitor = WindowMonitor()
    current_window = None
    try:
        while True:
            new_window = monitor.get_active_window()
            if new_window and new_window != current_window:
                current_window = new_window
                print(f"Active Window: {current_window}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
