"""Generates plot with current progress in training"""

import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

scratch_root = Path(os.environ.get("SCRATCH", "/scratch/dmillard"))
storage_root = Path(os.environ.get("DEF_ERA_STORAGE_ROOT", str(scratch_root / "def-era")))

# Path to your events file (override with TENSORBOARD_LOG_DIR as needed)
log_dir = os.environ.get(
    "TENSORBOARD_LOG_DIR",
    str(storage_root / "logs" / "lightning_logs" / "version_n"),
)
output_path = Path(os.environ.get("TRAIN_PROGRESS_FIGURE", str(storage_root / "outputs" / "train_results.png")))

# Initialize the event accumulator to read the TensorBoard logs
event_acc = EventAccumulator(log_dir)
event_acc.Reload()  # Load the events file

# Check available tags
tags = event_acc.Tags()["scalars"]
print("Available tags:", tags)

loss_data = event_acc.Scalars("train_loss_epoch")
val_data = event_acc.Scalars("val_loss_epoch")
epoch_data = event_acc.Scalars("epoch")

steps = [x.step for x in loss_data]  # Step (usually the global step)
loss_values = [x.value for x in loss_data]  # Loss values

# Plotting loss per step for training
plt.plot(steps, loss_values, label="Training")

steps = [x.step for x in val_data]
loss_values = [x.value for x in val_data]

# Plotting loss per step for validation
plt.plot(steps, loss_values, label="Validation")

plt.legend(frameon=False)
plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.title("Training Loss per Global Step")
plt.grid(True)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
