"""Module for playing back recorded Rocket League training data.

This script allows visualization of recorded gameplay data, including frames
and controller inputs, with playback controls for analysis and verification.
"""

import argparse
import glob
import os
from typing import Tuple

import cv2
import h5py
import numpy as np


def load_h5_data(filename: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load frames and inputs from an H5 file.

    Args:
        filename: Path to the H5 file to load.

    Returns:
        Tuple of (frames array, inputs array, metadata dictionary).
    """
    with h5py.File(filename, "r") as f:
        frames = f["frames"][:]
        inputs = f["inputs"][:]
        metadata = dict(f.attrs)
    return frames, inputs, metadata


def play_sequence(frames: np.ndarray, inputs: np.ndarray, metadata: dict, fps: int = 30) -> bool:
    """Play back the sequence of frames with corresponding inputs.

    Args:
        frames: Array of frames to display.
        inputs: Array of corresponding input values.
        metadata: Dictionary containing sequence metadata.
        fps: Frames per second for playback.

    Returns:
        False if user quit, True if sequence completed or skipped.
    """
    print("Sequence info:")
    print(f"Shape: {frames.shape}")
    print(f"Grayscale: {metadata.get('grayscale', 'unknown')}")
    print(f"Original target size: {metadata.get('target_size', 'unknown')}")

    for i, (frame, input_data) in enumerate(zip(frames, inputs)):
        # Convert grayscale to RGB for display if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Convert frame to uint8 if it's not already
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Add input data as text overlay
        frame_with_text = frame.copy()
        # cv2.putText(
        #     frame_with_text,
        #     input_text,
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        # )
        #
        # # Add frame counter
        # cv2.putText(
        #     frame_with_text,
        #     f"Frame: {i}/{len(frames)}",
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        # )

        # Display the frame
        cv2.imshow("Training Data Playback", frame_with_text)

        # Wait for the appropriate time to maintain FPS
        key = cv2.waitKey(1000 // fps) & 0xFF
        if key == ord("q"):
            return False  # Exit completely
        elif key == ord("n"):
            return True  # Skip to next sequence
        elif key == ord(" "):  # Space bar to pause
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(" "):  # Resume on space
                    break
                elif key == ord("q"):
                    return False
                elif key == ord("n"):
                    return True

    return True


def main() -> None:
    """Run the playback visualization tool.

    Loads and plays back recorded training sequences, providing controls
    for navigation (pause/resume, skip, quit). Handles command line arguments
    for data directory and playback speed.
    """
    parser = argparse.ArgumentParser(description="Playback H5 training data sequences")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="rlai-multi-map",
        help="Directory containing the training data",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for playback")
    args = parser.parse_args()

    # Get all H5 files
    pattern = os.path.join(args.data_dir, "*_batch.h5")
    h5_files = glob.glob(pattern)

    print(f"Found {len(h5_files)} H5 sequences")
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  N: Next sequence")
    print("  Q: Quit")

    for h5_file in sorted(h5_files):
        print(f"\nPlaying sequence: {os.path.basename(h5_file)}")
        try:
            frames, inputs, metadata = load_h5_data(h5_file)
            if not play_sequence(frames, inputs, metadata, args.fps):
                break  # Exit if user pressed 'q'
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error playing sequence {h5_file}: {e}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
