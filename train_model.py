import numpy as np
import cv2
import glob
import os
from typing import Tuple
import argparse


def load_data_pair(base_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the frames and inputs for a given base filename."""
    frames = np.load(f"{base_filename}_frames.npy")
    inputs = np.load(f"{base_filename}_inputs.npy")
    return frames, inputs


def play_sequence(frames: np.ndarray, inputs: np.ndarray, fps: int = 30):
    """Play back the sequence of frames with corresponding inputs."""
    for i, (frame, input_data) in enumerate(zip(frames, inputs)):
        # Convert frame to uint8 if it's not already
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Add input data as text overlay
        frame_with_text = frame.copy()
        input_text = f"Input: {input_data}"
        cv2.putText(
            frame_with_text,
            input_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Display the frame
        cv2.imshow("Training Data Playback", frame_with_text)

        # Wait for the appropriate time to maintain FPS
        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break


def main():
    parser = argparse.ArgumentParser(description="Playback training data sequences")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for playback"
    )
    args = parser.parse_args()

    # Get all unique base filenames (without _frames.npy or _inputs.npy)
    pattern = os.path.join(args.data_dir, "*_frames.npy")
    base_filenames = [f[:-11] for f in glob.glob(pattern)]  # Remove '_frames.npy'

    print(f"Found {len(base_filenames)} sequences")

    for base_filename in sorted(base_filenames):
        print(f"\nPlaying sequence: {os.path.basename(base_filename)}")
        try:
            frames, inputs = load_data_pair(base_filename)
            print(f"Sequence shape: {frames.shape}, Inputs shape: {inputs.shape}")
            play_sequence(frames, inputs, args.fps)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error playing sequence {base_filename}: {e}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
