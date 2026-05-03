"""
Simple utility to save training history and model info to a JSON file.
"""

import json
import os
from datetime import datetime
import torch


def save_run(path, model, history, **metadata):
    """
    Save training history and model info to a JSON file.

    Args:
        path (str): File path to save (e.g. "logs/run_v2.json").
        model (nn.Module): Trained model (used to record param count).
        history (dict): The dict returned by utils.trainer.fit().
        **metadata: Any extra info to store (e.g. epochs=50, lr=0.1, batch_size=128).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "class": type(model).__name__,
            "param_count": sum(p.numel() for p in model.parameters()),
        },
        "config": metadata,
        "history": history,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Run saved → {path}")
