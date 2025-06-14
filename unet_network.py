import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.Downsampling_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Downsampling_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Downsampling_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.Downsampling_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.last_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.Upsampling_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.Upsampl_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.Upsampling_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.Upsampl_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.Upsampling_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Upsampl_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Upsampling_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Upsampl_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 21, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, x):
        x = self.Downsampling_1(x)
        out1 = x
        x = self.maxpool(x)
        x = self.Downsampling_2(x)
        out2 = x
        x = self.maxpool(x)
        x = self.Downsampling_3(x)
        out3 = x
        x = self.maxpool(x)
        x = self.Downsampling_4(x)
        out4 = x
        x = self.maxpool(x)
        x = self.last_5(x)

        x = self.Upsampling_1(x)
        x = torch.cat([x, out4], dim=1)
        x = self.Upsampl_conv1(x)

        x = self.Upsampling_2(x)
        x = torch.cat([x, out3], dim=1)
        x = self.Upsampl_conv2(x)

        x = self.Upsampling_3(x)
        x = torch.cat([x, out2], dim=1)
        x = self.Upsampl_conv3(x)

        x = self.Upsampling_4(x)
        x = torch.cat([x, out1], dim=1)
        x = self.Upsampl_conv4(x)

        return x
