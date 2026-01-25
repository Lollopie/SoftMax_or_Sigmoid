import torch
from torch import nn


class UNetEncoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                         nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                         nn.Conv2d(channels, channels, kernel_size=(3, 3), padding='same'), nn.ReLU())
class UNetUpConv(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Upsample(scale_factor=2.0, mode='bilinear'),
                         nn.Conv2d(channels * 2, channels, kernel_size=(2, 2), padding='same'))
class UNetDecoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                       nn.Conv2d(channels, channels, kernel_size=(3, 3), padding='same'), nn.ReLU())

class UNet(nn.Module):
    def __init__(self, inputChannels, outLength):
        super().__init__()
        self.contracting = nn.ModuleList()
        currentChannels = 64
        self.input = nn.Sequential(nn.Conv2d(inputChannels, currentChannels, kernel_size=(3, 3), padding='same'), nn.ReLU(),
                         nn.Conv2d(currentChannels, currentChannels, kernel_size=(3, 3), padding='same'), nn.ReLU())
        self.encoding1 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding2 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding3 = UNetEncoding(currentChannels := currentChannels * 2)
        self.encoding4 = UNetEncoding(currentChannels := currentChannels * 2)
        self.upConv1 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding1 = UNetDecoding(currentChannels)
        self.upConv2 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding2 = UNetDecoding(currentChannels)
        self.upConv3 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding3 = UNetDecoding(currentChannels)
        self.upConv4 = UNetUpConv(currentChannels := currentChannels // 2)
        self.decoding4 = UNetDecoding(currentChannels)
        self.output = nn.Sequential(nn.Conv2d(currentChannels, outLength, kernel_size=(1, 1)))

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.encoding1(x1)
        x3 = self.encoding2(x2)
        x4 = self.encoding3(x3)
        x = self.encoding4(x4)
        x = self.upConv1(x)
        x = torch.cat((x4, x), dim=1)
        x = self.decoding1(x)
        x = self.upConv2(x)
        x = torch.cat((x3, x), dim=1)
        x = self.decoding2(x)
        x = self.upConv3(x)
        x = torch.cat((x2, x), dim=1)
        x = self.decoding3(x)
        x = self.upConv4(x)
        x = torch.cat((x1, x), dim=1)
        x = self.decoding4(x)
        x = self.output(x)
        return x

class DeepResidualUNet(nn.Module):
    def __init__(self, input_channels, out_length, base_channels = 64):
        # Add Convolutional Layers with Kernel 1 Stride 1 Padding same to increase channels of residual input
        super().__init__()
        self.encoding1 = nn.Sequential(nn.Conv2d(input_channels, base_channels, kernel_size=(3, 3), padding='same'),
                                       nn.BatchNorm2d(base_channels), nn.ReLU(),
                                       nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), padding='same'))
        self.upproject1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=(1, 1), stride=1, padding='same'),
            nn.BatchNorm2d(base_channels))
        current_channels = base_channels * 2
        self.encoding2 = nn.Sequential(nn.BatchNorm2d(base_channels), nn.ReLU(),
                                       nn.Conv2d(base_channels, current_channels, kernel_size=(3, 3), stride=2,
                                                 padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                 padding='same'))
        self.upproject2 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels *= 2
        self.encoding3 = nn.Sequential(nn.BatchNorm2d(current_channels // 2), nn.ReLU(),
                                       nn.Conv2d(current_channels // 2, current_channels, kernel_size=(3, 3), stride=2,
                                                 padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                 padding='same'))
        self.upproject3 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels *= 2
        self.bridge = nn.Sequential(nn.BatchNorm2d(current_channels // 2), nn.ReLU(),
                                    nn.Conv2d(current_channels // 2, current_channels, kernel_size=(3, 3), stride=2,
                                              padding=1), nn.BatchNorm2d(current_channels), nn.ReLU(),
                                    nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3), padding='same'))
        self.upproject4 = nn.Sequential(
            nn.Conv2d(current_channels // 2, current_channels, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding1 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject1 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding2 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject2 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        current_channels //= 2
        self.decoding3 = nn.Sequential(nn.BatchNorm2d(current_channels * 2 + current_channels), nn.ReLU(),
                                       nn.Conv2d(current_channels * 2 + current_channels, current_channels,
                                                 kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(current_channels),
                                       nn.ReLU(), nn.Conv2d(current_channels, current_channels, kernel_size=(3, 3),
                                                            padding='same'))
        self.downproject3 = nn.Sequential(
            nn.Conv2d(current_channels * 2 + current_channels, current_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(current_channels))
        self.output = nn.Conv2d(base_channels, out_length, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.encoding1(x) + self.upproject1(x)
        x2 = self.encoding2(x1) + self.upproject2(x1)
        x3 = self.encoding3(x2) + self.upproject3(x2)
        x = self.bridge(x3) + self.upproject4(x3)
        # Possibly Replace Upsample (and downproject) with ConvTranspose with Kernel and Stride 2
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x3, x), dim=1)
        x = self.decoding1(x) + self.downproject1(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x2, x), dim=1)
        x = self.decoding2(x) + self.downproject2(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = torch.cat((x1, x), dim=1)
        x = self.decoding3(x) + self.downproject3(x)
        x = self.output(x)
        return x
class AtrousDenseUDeconvNet(nn.Module):
    def __init__(self, inChannels, outChannels, dropoutChance=0.1):
        super().__init__()
        self.l2 = nn.BatchNorm2d(inChannels)
        self.l3 = nn.Sequential(nn.Conv2d(inChannels, 16, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=(3,3), padding='same', dilation=2), nn.ReLU())
        self.l5 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3,3), padding='same', dilation=5), nn.ReLU())
        self.l6 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3,3), padding='same', dilation=8), nn.ReLU())
        self.l8To11 = nn.Sequential(nn.Conv2d(72, 64, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                    nn.Dropout2d(dropoutChance),
                                    nn.AvgPool2d(kernel_size=(2,2)),
                                    nn.BatchNorm2d(64))
        self.l12 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(3,3), padding='same', dilation=2), nn.ReLU())
        self.l13 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3,3), padding='same', dilation=5), nn.ReLU())
        self.l14 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3,3), padding='same', dilation=8), nn.ReLU())
        self.l16To21 = nn.Sequential(nn.Conv2d(288, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance),
                                 nn.AvgPool2d(kernel_size=(2,2)),
                                 nn.BatchNorm2d(256),
                                 nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l22To24 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(128, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l25To27 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(256, 512, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance))
        self.l28To32 = nn.Sequential(nn.MaxPool2d(kernel_size=(2,2)),
                                 nn.Conv2d(512, 1024, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Dropout2d(dropoutChance),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(1024, 512, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l34To36 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(512, 256, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l38To40 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                 nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                 nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU())
        self.l42To46 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(64, 18, kernel_size=(3,3), stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(18, outChannels, kernel_size=(3,3), padding=1))
    def forward(self, x):
        x = self.l2(x)
        x1 = self.l3(x)
        x2 = self.l4(x1)
        x3 = self.l5(x2)
        x4 = self.l6(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = self.l8To11(x)
        x2 = self.l12(x1)
        x3 = self.l13(x2)
        x4 = self.l14(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = self.l16To21(x)
        x2 = self.l22To24(x1)
        x3 = self.l25To27(x2)
        x4 = self.l28To32(x3)
        x = torch.cat((x3, x4), dim=1)
        x = self.l34To36(x)
        x = torch.cat((x2, x), dim=1)
        x = self.l38To40(x)
        x = torch.cat((x1, x), dim=1)
        x = self.l42To46(x)
        return x