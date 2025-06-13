from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck

# from torchvision import models
#
# resnet50_pretrained = models.resnet50(pretrained=True)

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both conv1 and downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both conv2 and downsample layers downsample the input when stride != 1
#         self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
#         self.bn1 = norm_layer(width)
#         self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
#                                padding=dilation, groups=groups, bias=False, dilation=dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#     def forward(self, x):
#         return self._forward_impl(x)
#
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)/
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
# def resnet101(pretrained=True, progress=False, **kwargs):
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

print("----------------------------------------------------------------------------------")
import torch
import torch.nn.functional as F
import torch.nn as nn
# from torchvision.models import resnet101
import pretrained_network as pn
# model = resnet101(pretrained=True)
# modules = list(model.children())[:-2]  # 마지막 fc와 avgpool 레이어 제외
# resnet = nn.Sequential(*modules)
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
# import torch
# import torch.nn as nn
# import pretrained_network as pn
#
#
# class conv_bn_relu(nn.Module):
#     def __init__(self, c_in, c_out, kernel_size=3, pad=1):
#         super(conv_bn_relu, self).__init__()
#         self.cbr = nn.Sequential(
#             nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=pad),
#             nn.BatchNorm2d(c_out),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         return self.cbr(x)
#
# class PSPModule(nn.Module):
#     def __init__(self, c_in):
#         super(PSPModule, self).__init__()
#         self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.conv1 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)
#
#         self.avgpool2= nn.AdaptiveAvgPool2d(output_size=(2, 2))
#         self.conv2 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)
#
#         self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(3, 3))
#         self.conv3 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)
#
#         self.avgpool6 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
#         self.conv6 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)
#
#         self.conv_psp = nn.Conv2d(2 * c_in, c_in, kernel_size=1)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x1 = self.avgpool1(x)
#         x1 = self.conv1(x1)
#         x1 = torch.nn.functional.upsample_bilinear(x1, size=(x.shape[2], x.shape[3]))
#
#         x2 = self.avgpool2(x)
#         x2 = self.conv1(x2)
#         x2 = torch.nn.functional.upsample_bilinear(x2, size=(x.shape[2], x.shape[3]))
#
#         x3 = self.avgpool3(x)
#         x3 = self.conv1(x3)
#         x3 = torch.nn.functional.upsample_bilinear(x3, size=(x.shape[2], x.shape[3]))
#
#         x6 = self.avgpool6(x)
#         x6 = self.conv1(x6)
#         x6 = torch.nn.functional.upsample_bilinear(x6, size=(x.shape[2], x.shape[3]))
#
#         x = torch.cat([x, x1, x2, x3, x6], dim=1)
#         x = self.conv_psp(x)
#         x = self.relu(x)
#         return x
# class PSPNet(nn.Module):
#     def __init__(self, num_class):
#         super(PSPNet, self).__init__()
#         self.feats = pn.resnet50(pretrained=True)
#         self.psp = PSPModule(2048)
#         self.pred = nn.Conv2d(64, num_class, kernel_size=1, padding=0)
#
#         self.conv1 = conv_bn_relu(2048, 256)
#         self.conv2 = conv_bn_relu(256,128)
#         self.conv3 = conv_bn_relu(128,64)
#
#     def forward(self, x):
#         x = self.feats(x) # B 2048 28 28
#         x = self.psp(x) # B 2048 28 28
#
#
#         x = torch.nn.functional.upsample_bilinear(x, size=(x.shape[2], x.shape[3]))
#         x = self.conv1(x) # B 256 56 56
#
#         x = torch.nn.functional.upsample_bilinear(x, size=(x.shape[2], x.shape[3]))
#         x = self.conv2(x) # B 128 112 112
#
#         x = torch.nn.functional.upsample_bilinear(x, size=(x.shape[2], x.shape[3]))
#         x = self.conv3(x) # B 64 224 224
#         x = self.pred(x)
#         return x