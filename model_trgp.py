from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d


# Define specifc conv layer
class Conv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # define the scale v
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.0)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.0)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)
        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(
                torch.ones(self.out_channels).view(-1, 1, 1, 1) * 0.02,
                requires_grad=True,
            )
            self.alpha_w2 = nn.Parameter(
                torch.ones(self.out_channels).view(-1, 1, 1, 1) * 0.02,
                requires_grad=True,
            )

    def forward(self, input, space1=None, space2=None):

        if self.noise:
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight.clone().normal_(0, std)
        if space1 is not None or space2 is not None:
            sz = self.weight.data.size(0)

            if space2 is None:
                real_scale1 = self.scale1[: space1.size(1), : space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                # [chout, chinxkxk]  [chinxkxk, chinxkxk]
                proj_weight = torch.mm(self.weight.view(sz, -1), norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz, -1), torch.mm(space1, space1.transpose(1, 0))).view(
                    self.weight.size()
                )
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is None:

                real_scale2 = self.scale2[: space2.size(1), : space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))

                proj_weight = torch.mm(self.weight.view(sz, -1), norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz, -1), torch.mm(space2, space2.transpose(1, 0))).view(
                    self.weight.size()
                )

                if self.noise:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight
            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[: space1.size(1), : space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight.view(sz, -1), norm_project1).view(self.weight.size())
                diag_weight1 = torch.mm(self.weight.view(sz, -1), torch.mm(space1, space1.transpose(1, 0))).view(
                    self.weight.size()
                )

                real_scale2 = self.scale2[: space2.size(1), : space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight.view(sz, -1), norm_project2).view(self.weight.size())
                diag_weight2 = torch.mm(self.weight.view(sz, -1), torch.mm(space2, space2.transpose(1, 0))).view(
                    self.weight.size()
                )

                if self.noise:
                    masked_weight = (
                        proj_weight1
                        - diag_weight1
                        + proj_weight2
                        - diag_weight2
                        + self.weight
                        + ((self.alpha_w2 + self.alpha_w1) / 2) * noise * self.noise
                    )
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight

        else:
            masked_weight = self.weight

        return F.conv2d(
            input,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Define specific linear layer
class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

        # define the scale v
        scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        scale.fill_(0.0)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.0)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)
        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(torch.ones(self.weight.size()) * 0.1, requires_grad=True)
            self.alpha_w2 = nn.Parameter(torch.ones(self.weight.size()) * 0.1, requires_grad=True)

        # self.fixed_scale = scale

    def forward(self, input, space1=None, space2=None):
        if self.noise:
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight.clone().normal_(0, std)
        if space1 is not None or space2 is not None:
            sz = self.weight.data.size(0)

            if space2 is None:

                real_scale1 = self.scale1[: space1.size(1), : space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))

                proj_weight = torch.mm(self.weight, norm_project)

                diag_weight = torch.mm(self.weight, torch.mm(space1, space1.transpose(1, 0)))
                # masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is None:

                real_scale2 = self.scale2[: space2.size(1), : space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))

                proj_weight = torch.mm(self.weight, norm_project)
                diag_weight = torch.mm(self.weight, torch.mm(space2, space2.transpose(1, 0)))

                # masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[: space1.size(1), : space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight, norm_project1)
                diag_weight1 = torch.mm(self.weight, torch.mm(space1, space1.transpose(1, 0)))

                real_scale2 = self.scale2[: space2.size(1), : space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight, norm_project2)
                diag_weight2 = torch.mm(self.weight, torch.mm(space2, space2.transpose(1, 0)))

                # masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
                if self.noise and self.training:
                    masked_weight = (
                        proj_weight1
                        - diag_weight1
                        + proj_weight2
                        - diag_weight2
                        + self.weight
                        + ((self.alpha_w2 + self.alpha_w1) / 2) * noise * self.noise
                    )
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight

        else:
            masked_weight = self.weight
        return F.linear(input, masked_weight, self.bias)


# Define AlexNet model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class AlexNet(nn.Module):
    def __init__(self, taskcla):
        super(AlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

    def forward(
        self,
        x,
        space1=[None, None, None, None, None],
        space2=[None, None, None, None, None],
    ):
        bsz = deepcopy(x.size(0))
        if space1[0] is not None or space2[0] is not None:
            self.act["conv1"] = x
            x = self.conv1(x, space1=space1[0], space2=space2[0])
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            self.act["conv2"] = x
            x = self.conv2(x, space1=space1[1], space2=space2[1])
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            self.act["conv3"] = x
            x = self.conv3(x, space1=space1[2], space2=space2[2])
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            self.act["fc1"] = x
            x = self.fc1(x, space1=space1[3], space2=space2[3])
            x = self.drop2(self.relu(self.bn4(x)))

            self.act["fc2"] = x
            x = self.fc2(x, space1=space1[4], space2=space2[4])
            x = self.drop2(self.relu(self.bn5(x)))
            y = []
            for t, i in self.taskcla:
                y.append(self.fc3[t](x))
        else:
            self.act["conv1"] = x
            x = self.conv1(x)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            self.act["conv2"] = x
            x = self.conv2(x)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            self.act["conv3"] = x
            x = self.conv3(x)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x = x.view(bsz, -1)
            self.act["fc1"] = x
            x = self.fc1(x)
            x = self.drop2(self.relu(self.bn4(x)))

            self.act["fc2"] = x
            x = self.fc2(x)
            x = self.drop2(self.relu(self.bn5(x)))
            y = []
            for t, i in self.taskcla:
                y.append(self.fc3[t](x))

        return y

## Define ResNet18 model

def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv7x7(in_planes, out_planes, stride=1):
    return Conv2d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, space1=[None], space2=[None]):
        if space1[0] is not None or space2[0] is not None:
            self.count = self.count % 2
            self.act["conv_{}".format(self.count)] = x
            self.count += 1
            out = relu(self.bn1(self.conv1(x, space1=space1[0], space2=space2[0])))
            self.count = self.count % 2
            self.act["conv_{}".format(self.count)] = out
            self.count += 1
            out = self.bn2(self.conv2(out, space1=space1[1], space2=space2[1]))

            out += self.shortcut(x)

            out = relu(out)
        else:
            self.count = self.count % 2
            self.act["conv_{}".format(self.count)] = x
            self.count += 1
            out = relu(self.bn1(self.conv1(x)))
            self.count = self.count % 2
            self.act["conv_{}".format(self.count)] = out
            self.count += 1
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.taskcla = taskcla
        self.linear = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, space1=[None], space2=[None]):

        bsz = x.size(0)
        if space1[0] is not None or space2[0] is not None:
            self.act["conv_in"] = x.view(bsz, 3, 84, 84)
            out = relu(
                self.bn1(
                    self.conv1(
                        x.view(bsz, 3, 84, 84), space1=space1[0], space2=space2[0]
                    )
                )
            )

            out = self.layer1[0](out, space1=space1[1:3], space2=space2[1:3])
            out = self.layer1[1](out, space1=space1[3:5], space2=space2[3:5])
            out = self.layer2[0](out, space1=space1[5:8], space2=space2[5:8])
            out = self.layer2[1](out, space1=space1[8:10], space2=space2[8:10])
            out = self.layer3[0](out, space1=space1[10:13], space2=space2[10:13])
            out = self.layer3[1](out, space1=space1[13:15], space2=space2[13:15])
            out = self.layer4[0](out, space1=space1[15:18], space2=space2[15:18])
            out = self.layer4[1](out, space1=space1[18:20], space2=space2[18:20])

            # out = self.layer1(out, space1=space1[1:6], space2 = space2[1:6] )
            # out = self.layer2(out, space1=space1[6:10], space2 = space2[6:10])
            # out = self.layer3(out, space1=space1[10:14], space2 = space2[10:14])
            # out = self.layer4(out, space1=space1[14:19], space2 = space2[14:19])
            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y = []
            for t, i in self.taskcla:
                y.append(self.linear[t](out))
        else:
            # print(x.shape)
            self.act["conv_in"] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
            out = self.layer1(out)
            # print(out.size())
            out = self.layer2(out)
            # print(out.size())
            out = self.layer3(out)
            # print(out.size())
            out = self.layer4(out)
            # print(out.size())
            out = avg_pool2d(out, 2)
            # print(out.size())
            out = out.view(out.size(0), -1)
            # print(out.size())
            y = []
            for t, i in self.taskcla:
                y.append(self.linear[t](out))
        return y


def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)