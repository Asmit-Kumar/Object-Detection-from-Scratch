"""
Archived: residual single-object bounding-box regressor ("light" variant).

Originally `ObjectDetectorRes` (with its `ResBlock`), kept as commented-out code in
models/object_detector_res.py from commit 161ffb4 until 2292f24. One box per image, no classes.

Checkpoints: weights/plain_res_bbox_model_torch_light.pth,
checkpoint/{best_,checkpoint_}plain_res_bbox_model_torch_light.pth.pth.
Two related variants were never committed, so their checkpoints have no class here: the non-light one
(head input 4096: plain_res_bbox_model_torch.pth, res_bbox_model_torch.pth) and one with
squeeze-and-excitation blocks (a_bbox_model_torch.pth, a_od_checkpoint_light.pth).

    python -m models.archive.single_box_resnet
"""
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """Standard residual block with a skip connection (biased convs, 1x1 projection when widths differ)."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class SingleBoxResNet(nn.Module):
    """
    Residual bounding-box regressor: stem + three residual stages, each followed by a 2x max-pool,
    then a 4x4 adaptive pool and an FC head.

    Outputs (B, 4) box coordinates.
    """

    INPUT_SHAPE = (1, 128, 128)  # same canvas as SingleBoxCNN; the adaptive pool accepts any size

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.res1 = nn.Sequential(ResBlock(32, 64),  nn.MaxPool2d(2))
        self.res2 = nn.Sequential(ResBlock(64, 128), nn.MaxPool2d(2))
        self.res3 = nn.Sequential(ResBlock(128, 64), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # 128 -> 64 (stem) -> 32 (res1) -> 16 (res2) -> 8 (res3) -> 4x4 pool -> (B, 4)
        return self.head(self.res3(self.res2(self.res1(self.stem(x)))))


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.single_box_resnet")
