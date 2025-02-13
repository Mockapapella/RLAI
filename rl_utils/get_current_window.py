import subprocess
import time


def get_active_window():
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == "__main__":
    current_window = None
    try:
        while True:
            new_window = get_active_window()
            if new_window and new_window != current_window:
                current_window = new_window
                print(f"Active Window: {current_window}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
