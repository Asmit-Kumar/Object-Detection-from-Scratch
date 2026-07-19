import torch
from torch import nn
from torch.nn import functional as F


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ObjectDetectorResNet(nn.Module):
    """ResNet-style object detector with configurable channel widths.

    Args:
        max_objects (int):      Maximum number of predicted bounding boxes. Default: 24.
        channels (list[int]):   Channel widths for the 4 residual layers.
                                Use ``get_detector(size=...)`` for named presets.
        pool_size (int):        Spatial size for AdaptiveAvgPool2d before the FC head.
                                1 = global average pool (most compact).
                                2 = 2x2 spatial grid (retains coarse layout, default).
                                7 = original large pool (very high param count).
    """

    # Named size presets: (stem_out, [layer1..4 channels])
    CONFIGS = {
        "n": (32, [32,  64,  64,  128]),   # Nano   ~1.1M params
        "s": (64, [64,  128, 128, 256]),   # Small  ~3.0M params  (default / v1)
        "m": (64, [128, 256, 256, 512]),   # Medium ~9.2M params
    }

    def __init__(self, max_objects: int = 24, channels: list[int] | None = None, pool_size: int = 2):
        super().__init__()
        if channels is None:
            channels = self.CONFIGS["s"][1]
            stem_out = self.CONFIGS["s"][0]
        else:
            # stem_out = first channel entry in the list
            stem_out = channels[0]

        assert len(channels) == 4, "channels must have exactly 4 entries (one per res-layer)"

        self.max_objects = max_objects

        # Stem
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        # Residual layers — strides fixed at 2 to progressively downsample
        c = channels
        self.layer1 = SimpleResBlock(stem_out,  c[0], stride=2)
        self.layer2 = SimpleResBlock(c[0],      c[1], stride=2)
        self.layer3 = SimpleResBlock(c[1],      c[2], stride=2)
        self.layer4 = SimpleResBlock(c[2],      c[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Shared FC: flatten → shared representation → common hidden
        shared_dim = c[3] * 2          # e.g. 512 for Small
        head_dim   = shared_dim // 2   # e.g. 256 for Small
        fc_in      = c[3] * pool_size * pool_size
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, shared_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shared_dim, head_dim),    # common hidden layer
            nn.ReLU(inplace=True),
        )

        # Branch: thin final projections from the shared hidden representation
        self.box_head = nn.Linear(head_dim, max_objects * 4)   # [x, y, w, h]
        self.obj_head = nn.Linear(head_dim, max_objects * 1)   # confidence logit

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        shared = self.shared_fc(x)                              # (B, shared_dim)

        boxes = self.box_head(shared).view(-1, self.max_objects, 4)   # (B, N, 4)
        confs = self.obj_head(shared).view(-1, self.max_objects, 1)   # (B, N, 1)

        return torch.cat([boxes, confs], dim=-1)                      # (B, N, 5)


if __name__ == "__main__":
    for size in ("n", "s", "m"):
        stem, ch = ObjectDetectorResNet.CONFIGS[size]
        m = ObjectDetectorResNet(channels=ch)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {size}  channels={ch}  params={n:,}")

