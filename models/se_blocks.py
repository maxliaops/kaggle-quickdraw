from torch import nn


class SpatialChannelSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.cse = ChannelSEBlock(channel, reduction)
        self.sse = SpatialSEBlock(channel)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


class ChannelSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialSEBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return x * y
