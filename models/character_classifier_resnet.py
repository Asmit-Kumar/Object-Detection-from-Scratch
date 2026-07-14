from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))

class CharacterClassifierResNet(nn.Module):

    def __init__(self, num_classes=62):
        super(CharacterClassifierResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.block1 = BasicBlock(32, 64, stride=2)
        self.block2 = BasicBlock(64, 64, stride=2)
        self.block3 = BasicBlock(64, 64)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
            nn.Linear(64*3*3, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CharacterClassifierResNet(num_classes=62)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

