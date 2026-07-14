import torch
from torch import nn
from torch.nn import functional as F


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ObjectDetectorResNet(nn.Module):
    def __init__(self, max_objects=24):
        super(ObjectDetectorResNet, self).__init__()
        self.max_objects = max_objects
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = SimpleResBlock(64, 64, stride=2)
        self.layer2 = SimpleResBlock(64, 128, stride=2)
        self.layer3 = SimpleResBlock(128, 128, stride=2)
        self.layer4 = SimpleResBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, max_objects * 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, self.max_objects, 5)

        return x



# class ResBlock(nn.Module):
#     """Standard residual block with a skip connection."""
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
#             nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c)
#         )
#         self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
#
#     def forward(self, x):
#         return F.relu(self.conv(x) + self.skip(x))
#
#
# class ObjectDetectorRes(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
#         )
#         self.res1 = nn.Sequential(ResBlock(32, 64),  nn.MaxPool2d(2))
#         self.res2 = nn.Sequential(ResBlock(64, 128), nn.MaxPool2d(2))
#         self.res3 = nn.Sequential(ResBlock(128, 64), nn.MaxPool2d(2))
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((4, 4)),
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 4)
#         )
#
#     def forward(self, x):
#         return self.head(self.res3(self.res2(self.res1(self.stem(x)))))

if __name__ == '__main__':
    model = ObjectDetectorResNet()
    print(model)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
