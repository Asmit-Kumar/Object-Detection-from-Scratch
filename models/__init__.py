"""
Shared model definitions for the Object Detection & Character Recognition pipeline.
Import these instead of re-defining the architectures in every script.
"""

import torch

from .object_detector_res import ObjectDetectorResNet
from .character_classifier_resnet import CharacterClassifierResNet

def load_bbox_model(path, device):
    """Load ObjectDetectorResNet weights from path onto device."""
    model = ObjectDetectorResNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_character_model(path, device, num_classes: int = 62):
    """Load CharacterClassifierResNet weights from path onto device."""
    model = CharacterClassifierResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
