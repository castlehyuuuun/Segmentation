import torch
import torch.nn as nn
import torch.nn.functional as F

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
#         self.bn5 = nn.BatchNorm2d(256)
#
#         self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
#         self.bn6 = nn.BatchNorm2d(256)
#
#         self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
#         self.bn7 = nn.BatchNorm2d(512)
#
#         self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
#         self.bn8 = nn.BatchNorm2d(512)
#
#         self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=True)
#         self.bn9 = nn.BatchNorm2d(1024)
#
#         self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=True)
#         self.bn10 = nn.BatchNorm2d(1024)
#         '''
#         upsampling
#         '''
#         self.conv11 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0, bias=True)
#         self.bn11 = nn.BatchNorm2d(512)
#
#         self.conv12 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, bias=True)
#         self.bn12 = nn.BatchNorm2d(512)
#
#         self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
#         self.bn13 = nn.BatchNorm2d(512)
#
#         self.conv14 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, bias=True)
#         self.bn14 = nn.BatchNorm2d(256)
#
#         self.conv15 = nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True)
#         self.bn15 = nn.BatchNorm2d(256)
#
#         self.conv16 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
#         self.bn16 = nn.BatchNorm2d(256)
#
#         self.conv17 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, bias=True)
#         self.bn17 = nn.BatchNorm2d(128)
#
#         self.conv18 = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)
#         self.bn18 = nn.BatchNorm2d(128)
#
#         self.conv19 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)
#         self.bn19 = nn.BatchNorm2d(128)
#
#         self.conv20 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, bias=True)
#         self.bn20 = nn.BatchNorm2d(64)
#
#         self.conv21 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
#         self.bn21 = nn.BatchNorm2d(64)
#
#         self.conv22 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
#         self.bn22 = nn.BatchNorm2d(64)
#
#         self.conv23 = nn.Conv2d(64, 21, 1, stride=1, padding=0, bias=True)
#
#
#
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out1 = out
#         out = self.maxpool(out)
#
#         out = F.relu(self.bn3(self.conv3(out)))
#         out = F.relu(self.bn4(self.conv4(out)))
#         out2 = out
#         out = self.maxpool(out)
#
#         out = F.relu(self.bn5(self.conv5(out)))
#         out = F.relu(self.bn6(self.conv6(out)))
#         out3 = out
#         out = self.maxpool(out)
#
#         out = F.relu(self.bn7(self.conv7(out)))
#         out = F.relu(self.bn8(self.conv8(out)))
#         out4 = out
#         out = self.maxpool(out)
#
#         out = F.relu(self.bn9(self.conv9(out)))
#         out = F.relu(self.bn10(self.conv10(out))) # last downsampling
#
#         out = F.relu(self.bn11(self.conv11(out)))
#         out = torch.cat((out4, out), dim=1)
#
#         out = F.relu(self.bn12(self.conv12(out)))
#         out = F.relu(self.bn13(self.conv13(out)))
#         out = F.relu(self.bn14(self.conv14(out)))
#         out = torch.cat((out3, out), dim=1)
#
#         out = F.relu(self.bn15(self.conv15(out)))
#         out = F.relu(self.bn16(self.conv16(out)))
#         out = F.relu(self.bn17(self.conv17(out)))
#         out = torch.cat((out2, out), dim=1)
#
#         out = F.relu(self.bn18(self.conv18(out)))
#         out = F.relu(self.bn19(self.conv19(out)))
#         out = F.relu(self.bn20(self.conv20(out)))
#         out = torch.cat((out1, out), dim=1)
#
#         out = F.relu(self.bn21(self.conv21(out)))
#         out = F.relu(self.bn22(self.conv22(out)))
#         out = self.conv23(out)
#         out = F.softmax(out, dim=1)
#         return out



# class UNet(nn.Module):
#     def __init__(self):
#
#         super(UNet, self).__init__()
#
#         def DBconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#             layers = []
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                  kernel_size=kernel_size, stride=stride, padding=padding,
#                                  bias=bias)]
#             layers += [nn.BatchNorm2d(num_features=out_channels)]
#             layers += [nn.ReLU()]
#
#             dbconv = nn.Sequential(*layers)
#
#             return dbconv
#
#         # Contracting path
#         self.enc1_1 = DBconv(in_channels=3, out_channels=64)
#         self.enc1_2 = DBconv(in_channels=64, out_channels=64)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc2_1 = DBconv(in_channels=64, out_channels=128)
#         self.enc2_2 = DBconv(in_channels=128, out_channels=128)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc3_1 = DBconv(in_channels=128, out_channels=256)
#         self.enc3_2 = DBconv(in_channels=256, out_channels=256)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc4_1 = DBconv(in_channels=256, out_channels=512)
#         self.enc4_2 = DBconv(in_channels=512, out_channels=512)
#
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc5_1 = DBconv(in_channels=512, out_channels=1024)
#
#         # Expansive path
#         self.dec5_1 = DBconv(in_channels=1024, out_channels=512)
#
#         self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec4_2 = DBconv(in_channels=2 * 512, out_channels=512)
#         self.dec4_1 = DBconv(in_channels=512, out_channels=256)
#
#         self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec3_2 = DBconv(in_channels=2 * 256, out_channels=256)
#         self.dec3_1 = DBconv(in_channels=256, out_channels=128)
#
#         self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec2_2 = DBconv(in_channels=2 * 128, out_channels=128)
#         self.dec2_1 = DBconv(in_channels=128, out_channels=64)
#
#         self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec1_2 = DBconv(in_channels=2 * 64, out_channels=64)
#         self.dec1_1 = DBconv(in_channels=64, out_channels=64)
#
#         self.fc = nn.Conv2d(in_channels=64, out_channels=21, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x):
#         enc1_1 = self.enc1_1(x)
#         enc1_2 = self.enc1_2(enc1_1)
#         pool1 = self.pool1(enc1_2)
#
#         enc2_1 = self.enc2_1(pool1)
#         enc2_2 = self.enc2_2(enc2_1)
#         pool2 = self.pool2(enc2_2)
#
#         enc3_1 = self.enc3_1(pool2)
#         enc3_2 = self.enc3_2(enc3_1)
#         pool3 = self.pool3(enc3_2)
#
#         enc4_1 = self.enc4_1(pool3)
#         enc4_2 = self.enc4_2(enc4_1)
#         pool4 = self.pool4(enc4_2)
#
#         enc5_1 = self.enc5_1(pool4)
#
#         dec5_1 = self.dec5_1(enc5_1)
#
#         unpool4 = self.unpool4(dec5_1)
#         cat4 = torch.cat([unpool4, enc4_2], dim=1)
#         dec4_2 = self.dec4_2(cat4)
#         dec4_1 = self.dec4_1(dec4_2)
#
#         unpool3 = self.unpool3(dec4_1)
#         cat3 = torch.cat([unpool3, enc3_2], dim=1)
#         dec3_2 = self.dec3_2(cat3)
#         dec3_1 = self.dec3_1(dec3_2)
#
#         unpool2 = self.unpool2(dec3_1)
#         cat2 = torch.cat([unpool2, enc2_2], dim=1)
#         dec2_2 = self.dec2_2(cat2)
#         dec2_1 = self.dec2_1(dec2_2)
#
#         unpool1 = self.unpool1(dec2_1)
#         cat1 = torch.cat([unpool1, enc1_2], dim=1)
#         dec1_2 = self.dec1_2(cat1)
#         dec1_1 = self.dec1_1(dec1_2)
#
#         x = self.fc(dec1_1)
#
#         return x

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