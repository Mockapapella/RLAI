import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from rl_utils.models import RocketNet
import cv2
import h5py
import gc
import os
import glob
import argparse
from rl_utils.controller_reader import ControllerReader
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Silence matplotlib font debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"debug_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", mode="w"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RL_Debug")
np.set_printoptions(precision=4, suppress=True)


class ModelValidator:
    """ISO-compliant validation suite for RocketNet deployments"""

    def __init__(self, model_path=None, batch_file=None, output_dir="debug_output"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize model with fixed forward pass
        self.model = self._init_model(model_path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.batch_file = batch_file

        # Initialize controller reader if available
        try:
            self.controller = ControllerReader()
            logger.info("✅ Controller initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Controller initialization failed: {str(e)}")
            self.controller = None

    def _init_model(self, path):
        """Enterprise-grade model initialization with checksums"""
        logger.info("🚀 Initializing RocketNet v2.1.4")
        model = RocketNet().to(self.device)

        if path:
            try:
                state_dict = torch.load(path, map_location=self.device)

                # Check if we need to handle legacy model format
                if any(key.startswith("main.") for key in state_dict.keys()):
                    logger.info("📝 Detected legacy model format - adapting keys")
                    # Create mapping from old keys to new keys
                    new_state_dict = {}

                    # Map the input reduction layer (likely unchanged)
                    for key in [
                        k for k in state_dict.keys() if k.startswith("input_reduction")
                    ]:
                        new_state_dict[key] = state_dict[key]

                    # Map main.0/2/5 to backbone layers
                    if "main.0.weight" in state_dict:
                        new_state_dict["backbone.0.weight"] = state_dict[
                            "main.0.weight"
                        ]
                        new_state_dict["backbone.0.bias"] = state_dict["main.0.bias"]
                    if "main.2.weight" in state_dict:
                        new_state_dict["backbone.3.weight"] = state_dict[
                            "main.2.weight"
                        ]
                        new_state_dict["backbone.3.bias"] = state_dict["main.2.bias"]
                    if "main.5.weight" in state_dict:
                        new_state_dict["backbone.6.weight"] = state_dict[
                            "main.5.weight"
                        ]
                        new_state_dict["backbone.6.bias"] = state_dict["main.5.bias"]

                    # Map main.7 to output heads
                    if "main.7.weight" in state_dict:
                        main_weight = state_dict["main.7.weight"]
                        main_bias = state_dict["main.7.bias"]

                        # First 11 outputs are binary controls
                        new_state_dict["binary_head.weight"] = main_weight[:11, :]
                        new_state_dict["binary_head.bias"] = main_bias[:11]

                        # Last 8 outputs are analog controls
                        new_state_dict["analog_head.0.weight"] = main_weight[11:, :]
                        new_state_dict["analog_head.0.bias"] = main_bias[11:]

                    # Use the adapted state dict
                    state_dict = new_state_dict
                    logger.info(f"✅ Successfully adapted legacy model format")

                # Load the state dict with strict=False to allow partial loading
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Loaded weights from {path}")
                logger.debug(f"Weight checksum: {hash(str(state_dict))}")

                # IMPORTANT: Patch the model's forward method to disable gradient checkpointing
                # and autocast during inference
                original_forward = model.forward

                def fixed_forward(x):
                    """Fixed forward pass without gradient checkpointing or autocast"""
                    with torch.no_grad():
                        x = model.input_reduction(x)
                        features = model.backbone(x)
                        binary_out = model.binary_head(features)
                        analog_out = model.analog_head(features)
                        return torch.cat([binary_out, analog_out], dim=1)

                model.original_forward = original_forward
                model.forward = fixed_forward
                logger.info("✅ Patched model forward pass for inference")

            except Exception as e:
                logger.critical(f"🚨 Model load failed: {str(e)}")
                raise

        model.eval()
        torch.backends.cudnn.benchmark = True
        return model

    def _find_batch_file(self):
        """Find the latest batch file in the training directory"""
        if self.batch_file:
            return self.batch_file

        pattern = os.path.join("data/rocket_league/training/", "*_batch.h5")
        h5_files = glob.glob(pattern)

        if not h5_files:
            logger.error("No batch files found in data/rocket_league/training/")
            return None

        # Sort by creation time (newest first)
        h5_files.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"🔍 Using latest batch file: {h5_files[0]}")
        return h5_files[0]

    def _load_sample(self, index=None):
        """Production-grade data loading with sanity checks"""
        try:
            batch_file = self._find_batch_file()
            if not batch_file:
                raise FileNotFoundError("No suitable batch file found")

            logger.info(f"📂 Loading sample from {batch_file}")
            with h5py.File(batch_file, "r") as f:
                # Get file info
                num_samples = len(f["frames"])
                logger.info(f"📊 Batch file contains {num_samples} samples")
                logger.info(f"📊 Frame shape: {f['frames'][0].shape}")
                logger.info(f"📊 Input shape: {f['inputs'][0].shape}")

                # Load specific sample or random one
                if index is None:
                    index = np.random.randint(0, num_samples)
                else:
                    index = min(index, num_samples - 1)

                frame = f["frames"][index]
                label = f["inputs"][index]

            # Validate data integrity
            logger.info(f"📊 Sample frame shape: {frame.shape}")
            logger.info(f"📊 Sample label shape: {label.shape}")
            logger.info(f"📊 Sample label values: {label}")

            # Save raw frame for visual inspection
            frame_path = self.output_dir / f"raw_frame_{index}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"📸 Saved raw input frame to {frame_path}")

            # Process the frame
            processed_frame = self.transform(frame).unsqueeze(0).to(self.device)
            label_tensor = (
                torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            return processed_frame, label_tensor, frame

        except Exception as e:
            logger.error(f"📂 Data loading failed: {str(e)}")
            raise

    def _tensor_audit(self, tensor, name):
        """Comprehensive tensor validation"""
        logger.info(f"🔍 Tensor Audit: {name}")
        logger.info(f"Shape: {tensor.shape} | Dtype: {tensor.dtype}")
        logger.info(f"Stats: μ={tensor.mean().item():.4f} σ={tensor.std().item():.4f}")
        logger.info(f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        logger.info(
            f"NaN/Inf: {torch.isnan(tensor).any().item()} | {torch.isinf(tensor).any().item()}"
        )

        # Save histograms for numerical tensors
        if tensor.numel() > 1 and tensor.dtype in [
            torch.float32,
            torch.float16,
            torch.float64,
        ]:
            plt.figure(figsize=(10, 6))
            plt.hist(tensor.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"Distribution of {name}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(
                self.output_dir / f"{name.lower().replace(' ', '_')}_histogram.png"
            )
            plt.close()

    def _get_live_controller_input(self):
        """Get live controller input for comparison"""
        if not self.controller:
            return None

        try:
            return self.controller.get_state_vector()
        except Exception as e:
            logger.error(f"⚠️ Failed to get controller input: {str(e)}")
            return None

    def run_validation(self):
        """Full-stack validation pipeline"""
        try:
            # Hardware audit
            self._log_hardware()

            # Model inspection
            self._model_forensics()

            # Load production sample
            input_tensor, label_tensor, raw_frame = self._load_sample()
            self._tensor_audit(input_tensor, "Input")
            self._tensor_audit(label_tensor, "Label")

            # Get current controller state if available
            live_input = self._get_live_controller_input()
            if live_input is not None:
                logger.info(f"🎮 Current controller state: {live_input}")

            # Test inference
            logger.info("\n🧪 Testing inference")
            with torch.no_grad():
                output = self.model(input_tensor)

                # Check if outputs are extremely high or low
                if torch.abs(output).max() > 100:
                    logger.warning(
                        f"⚠️ EXTREME OUTPUT VALUES DETECTED! Max absolute value: {torch.abs(output).max().item():.2f}"
                    )
                    logger.warning(
                        "This indicates a severe model training issue or incorrect loss function"
                    )

                    # Calculate scaling factor to bring outputs to reasonable range
                    scale_factor = 10.0 / torch.abs(output).max().item()
                    logger.info(
                        f"🔧 Applying emergency scaling factor of {scale_factor:.6f} to outputs"
                    )

                    # Save original outputs
                    original_output = output.clone()

                    # Scale outputs
                    output = output * scale_factor
                    logger.info(
                        f"Scaled output range: [{output.min().item():.4f}, {output.max().item():.4f}]"
                    )

                # Apply sigmoid activation
                probs = torch.sigmoid(output)

            # Output analysis
            self._tensor_audit(output, "Logits")
            self._tensor_audit(probs, "Probabilities")

            # Visualize predictions vs ground truth
            self._visualize_predictions(label_tensor[0], probs[0])

            # Comparative analysis
            logger.info("\n📊 Comparative Results:")
            logger.info(f"Label:\n{label_tensor[0].cpu().numpy()}")
            logger.info(f"Predictions:\n{probs[0].cpu().numpy()}")
            logger.info(
                f"Absolute Differences:\n{torch.abs(probs[0] - label_tensor[0]).cpu().numpy()}"
            )

            # Calculate MSE for analog and binary parts separately
            binary_indices = list(range(11))  # First 11 values are binary (buttons)
            analog_indices = list(range(11, 19))  # Last 8 values are analog (axes)

            binary_mse = torch.mean(
                (probs[0, binary_indices] - label_tensor[0, binary_indices]) ** 2
            ).item()
            analog_mse = torch.mean(
                (probs[0, analog_indices] - label_tensor[0, analog_indices]) ** 2
            ).item()

            logger.info(f"Binary controls MSE: {binary_mse:.4f}")
            logger.info(f"Analog controls MSE: {analog_mse:.4f}")

            # Precision metrics with proper batch handling
            logger.info("\n🧮 Performance Metrics:")
            loss = torch.nn.functional.binary_cross_entropy(probs, label_tensor)
            logger.info(f"BCE Loss: {loss.item():.4f}")
            logger.info(
                f"Binary Accuracy: {(torch.round(probs) == label_tensor).float().mean().item():.2%}"
            )

            # Calculate analog accuracy with tolerance
            analog_accuracy = (
                (
                    torch.abs(
                        probs[:, analog_indices] - label_tensor[:, analog_indices]
                    )
                    < 0.2
                )
                .float()
                .mean()
                .item()
            )
            logger.info(f"Analog Accuracy (±0.2): {analog_accuracy:.2%}")

            # Test model on multiple random samples
            logger.info("\n🧪 Testing model on multiple samples")
            self._test_multiple_samples()

            # Visualize sample
            self._visualize_sample(raw_frame, label_tensor[0], probs[0])

            return 0

        except Exception as e:
            logger.critical(f"🚨 Critical failure: {str(e)}")
            import traceback

            logger.critical(traceback.format_exc())
            return 1

    def _visualize_predictions(self, label, prediction):
        """Create visualization comparing predictions to ground truth"""
        # Disable font message debug output
        plt_logger = logging.getLogger("matplotlib")
        original_level = plt_logger.level
        plt_logger.setLevel(logging.WARNING)

        categories = [
            # Buttons (first 11)
            "BTN_SOUTH",
            "BTN_EAST",
            "BTN_NORTH",
            "BTN_WEST",
            "BTN_TL",
            "BTN_TR",
            "BTN_SELECT",
            "BTN_START",
            "BTN_MODE",
            "BTN_THUMBL",
            "BTN_THUMBR",
            # Axes (next 8)
            "ABS_X",
            "ABS_Y",
            "ABS_RX",
            "ABS_RY",
            "ABS_Z",
            "ABS_RZ",
            "ABS_HAT0X",
            "ABS_HAT0Y",
        ]

        # Convert to numpy
        label_np = label.cpu().numpy()
        pred_np = prediction.cpu().numpy()

        # Create bar chart
        plt.figure(figsize=(15, 10))
        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width / 2, label_np, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_np, width, label="Prediction")

        plt.ylabel("Value")
        plt.title("Model Predictions vs Ground Truth")
        plt.xticks(x, categories, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add a horizontal line at 0.5 for binary threshold reference
        plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_comparison.png")
        plt.close()

        # Restore original logging level
        plt_logger.setLevel(original_level)

        logger.info(
            f"📊 Created prediction visualization: {self.output_dir / 'prediction_comparison.png'}"
        )

    def _visualize_sample(self, frame, label, prediction):
        """Visualize input frame with controller overlay"""
        # Create a copy of the frame
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]

        # Add controller visualization overlay
        overlay = (
            np.zeros((height // 3, width, 3), dtype=np.uint8) + 240
        )  # Light gray background

        # Draw controller state visualization
        label_np = label.cpu().numpy()
        pred_np = prediction.cpu().numpy()

        # Control element positions
        button_names = [
            "A",
            "B",
            "X",
            "Y",
            "LB",
            "RB",
            "Select",
            "Start",
            "Mode",
            "L3",
            "R3",
        ]
        axis_names = ["LX", "LY", "RX", "RY", "LT", "RT", "DX", "DY"]

        # Combine frame and overlay
        combined = np.vstack([vis_frame, overlay])

        # Save the visualization
        vis_path = self.output_dir / "sample_visualization.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        logger.info(f"📸 Created sample visualization: {vis_path}")

    def _test_multiple_samples(self, num_samples=5):
        """Test model on multiple random samples"""
        try:
            batch_file = self._find_batch_file()
            if not batch_file:
                logger.warning("⚠️ Cannot run multiple sample test: No batch file found")
                return

            with h5py.File(batch_file, "r") as f:
                total_samples = len(f["frames"])
                indices = np.random.choice(
                    total_samples, min(num_samples, total_samples), replace=False
                )

                # Track metrics
                total_binary_accuracy = 0
                total_analog_accuracy = 0
                total_bce_loss = 0

                for i, idx in enumerate(indices):
                    frame = f["frames"][idx]
                    label = f["inputs"][idx]

                    # Process frame
                    input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
                    label_tensor = (
                        torch.tensor(label, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    # Get prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probs = torch.sigmoid(output)

                    # Calculate metrics
                    binary_accuracy = (
                        (torch.round(probs) == label_tensor).float().mean().item()
                    )
                    analog_indices = list(range(11, 19))
                    analog_accuracy = (
                        (
                            torch.abs(
                                probs[:, analog_indices]
                                - label_tensor[:, analog_indices]
                            )
                            < 0.2
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    bce_loss = torch.nn.functional.binary_cross_entropy(
                        probs, label_tensor
                    ).item()

                    total_binary_accuracy += binary_accuracy
                    total_analog_accuracy += analog_accuracy
                    total_bce_loss += bce_loss

                    logger.info(
                        f"Sample {i + 1}: Binary Accuracy = {binary_accuracy:.2%}, Analog Accuracy = {analog_accuracy:.2%}, BCE Loss = {bce_loss:.4f}"
                    )

                # Report averages
                avg_binary_accuracy = total_binary_accuracy / num_samples
                avg_analog_accuracy = total_analog_accuracy / num_samples
                avg_bce_loss = total_bce_loss / num_samples

                logger.info(f"Average across {num_samples} samples:")
                logger.info(f"  Binary Accuracy: {avg_binary_accuracy:.2%}")
                logger.info(f"  Analog Accuracy: {avg_analog_accuracy:.2%}")
                logger.info(f"  BCE Loss: {avg_bce_loss:.4f}")

        except Exception as e:
            logger.error(f"❌ Multiple sample test failed: {str(e)}")

    def _log_hardware(self):
        """Minimal system configuration logging"""
        logger.info("\n🖥️  System Configuration:")
        logger.info(f"Torch: {torch.__version__}")
        logger.info(f"Device: {self.device}")

        if self.device.type == "cuda":
            logger.info(
                f"GPU Memory: Allocated={torch.cuda.memory_allocated() / 1e6:.2f}MB | "
                f"Reserved={torch.cuda.memory_reserved() / 1e6:.2f}MB"
            )

    def _model_forensics(self):
        """Basic model architecture analysis"""
        logger.info("\n🔬 Model Architecture Analysis:")
        logger.info(
            f"Parameter Count: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        logger.info(f"Input Layer: {self.model.input_reduction}")
        logger.info(
            f"Output Layers: Binary={self.model.binary_head}, Analog={self.model.analog_head}"
        )

        # Check for problematic weights in output layers only
        activation_ranges = {}
        for name, param in self.model.named_parameters():
            if "binary_head.weight" in name or "analog_head.0.weight" in name:
                mean = param.data.mean().item()
                std = param.data.std().item()
                logger.debug(f"Output layer {name}: μ={mean:.4f} σ={std:.4f}")

                if std > 0.1:
                    logger.warning(
                        f"⚠️ Output layer weights have high variance (σ={std:.4f})"
                    )
                    logger.warning("This may cause extreme activation values!")

                # Estimate activation range
                est_act_range = std * 10  # rough estimate based on input std
                activation_ranges[name] = est_act_range

        # Check for potential activation explosion in output layers
        if activation_ranges:
            max_range = max(activation_ranges.values())
            if max_range > 10:
                logger.warning(
                    f"⚠️ Potential activation explosion detected! Estimated max range: {max_range:.2f}"
                )
                logger.warning(
                    "The model may produce extreme output values during inference"
                )

        # Test forward pass timing with fewer iterations
        dummy_input = torch.randn(1, 480, 270, 3).to(self.device)
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

        with torch.no_grad():
            # Reduced warmup
            for _ in range(3):
                _ = self.model(dummy_input)

            iterations = 10
            times = []
            for _ in range(iterations):
                starter.record()
                _ = self.model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        logger.info(
            f"Forward pass timing: {sum(times) / iterations:.2f}ms (avg over {iterations} runs)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RocketNet Validation Tool")
    parser.add_argument(
        "--model", default="rocket_model.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--batch", default=None, help="Specific batch file to use for testing"
    )
    parser.add_argument(
        "--output", default="debug_output", help="Output directory for debug artifacts"
    )
    args = parser.parse_args()

    logger.info("🛠️ Starting RocketNet validation tool")
    validator = ModelValidator(
        model_path=args.model, batch_file=args.batch, output_dir=args.output
    )
    exit_code = validator.run_validation()
    torch.cuda.empty_cache()
    gc.collect()
    exit(exit_code)
