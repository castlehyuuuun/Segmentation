from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.nn.functional as F
import torch.nn as nn

import pretrained_network as pn

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = pn.resnet50(pretrained=True)
        self.backbone1 = nn.Sequential(*list(self.model.children())[:-2])
        self.aux_classifier = nn.Conv2d(1024, 21, kernel_size=3, stride=1, padding=1)  # 1024는 layer3의 출력 채널 수

    def forward(self, x):
        x = self.backbone1[:6](x)
        aux_feature = self.backbone1[6](x)
        x = self.backbone1[7:](aux_feature)

        aux_out = self.aux_classifier(aux_feature)

        return x, aux_out

class cbr(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super(cbr, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cbr(x)

class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        self.backbone2 = resnet()
        # self.midChannel = int(inChannel / midReduction)  # 1x1Conv channel num, defalut=512
        self.num_pool = [1, 2, 3, 6]
        self.ppm = []
        self.ppm = nn.ModuleList([
            self.pyramid_pooling_block(2048, pool_size) for pool_size in self.num_pool
        ])
        self.conv_psp = nn.Conv2d(2 * 2048, 2048, kernel_size=1)
        self.bottleneck = nn.Conv2d(64, 21, kernel_size=1, padding=0)

        self.conv1 = cbr(2048, 256)
        self.conv2 = cbr(256, 128)
        self.conv3 = cbr(128, 64)

    def pyramid_pooling_block(self, in_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        )

    def forward(self, x):
        x, aux_out = self.backbone2(x)
        h, w = x.shape[2], x.shape[3]

        ppm_outs = []

        for pool in self.ppm:
            pooled_output = pool(x)

            upsampled_output = torch.nn.functional.upsample_bilinear(pooled_output, size=(h, w))

            ppm_outs.append(upsampled_output)
        x = torch.cat(ppm_outs + [x], dim=1)
        out = self.conv_psp(x)
        out = F.relu(out)

        out = torch.nn.functional.upsample_bilinear(out, size=(2 * h, 2 * w))
        out = self.conv1(out)
        out = torch.nn.functional.upsample_bilinear(out, size=(4 * h, 4 * w))
        out = self.conv2(out)
        out = torch.nn.functional.upsample_bilinear(out, size=(8 * h, 8 * w))
        out = self.conv3(out)
        out = self.bottleneck(out)
        if self.training:
            return out, aux_out
        else:
            return out


print("----------------------------------------------------------------------------------")
