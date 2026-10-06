"""
Archived: ultra-light 10-class digit classifier, paired with the single-box regressors.

Originally `DigitClassifier` in the root `models.py` (commit 788cc42, removed in 161ffb4), trained in
`DigitClassifier.ipynb`. Replaced by CharacterClassifierResNet (47/62 EMNIST classes).

Checkpoints: none of the files under checkpoint/ or weights/ match it.

    python -m models.archive.digit_classifier_cnn
"""
from torch import nn


class DigitClassifierCNN(nn.Module):
    """
    Ultra-light Digit Classifier (~130k params): 3-stage CNN backbone compressing to 3x3 before the head.

    Outputs raw logits for 10 digit classes.
    """

    INPUT_SHAPE = (1, 28, 28)  # MNIST-sized digit crop

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 3x3
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))   # (B, 10)


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.digit_cnn")
