import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


activations = {"none": nn.Identity, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


# https://github.com/alexsax/midlevel-reps/blob/visualpriors/visualpriors/taskonomy_network.py


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, bias=False, padding=1
        )
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = F.pad(out, pad=(1,1,1,1), mode='constant', value=0)  # other modes are reflect, replicate
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# For making a (Hp x Wp x 2048) feature map commmon for all tasks
class Backbone(nn.Module):
    def __init__(self, lightweight=True, layers=None):
        # def __init__(self, normalize_outputs=True, eval_only=True, train_penultimate=False, train=False):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        block = Bottleneck
        if layers is None:
            if lightweight:
                layers = [2, 2, 2, 2]
            else:
                layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []

        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.inplanes, planes, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(nn.MaxPool2d(kernel_size=1, stride=stride),)
        layers.append(block(self.inplanes, planes, stride, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, pad=(3, 3, 3, 3), mode="constant", value=0)
        #  other modes are reflect, replicate, constant

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.pad(x, (0,1,0,1), 'constant', 0)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Transform the (Hp x Wp x 2048) feature map to a task-specific feature map of (Hp x Wp x 8)
class Midreps(nn.Module):
    def __init__(self, normalize_outputs=True):
        super().__init__()
        self.compress1 = nn.Conv2d(
            2048, 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.compress_bn = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.normalize_outputs = normalize_outputs
        if self.normalize_outputs:
            self.groupnorm = nn.GroupNorm(8, 8, affine=False)

    def forward(self, x):
        x = self.compress1(x)
        x = self.compress_bn(x)
        x = self.relu1(x)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x


class Scissor(torch.nn.Module):
    # Remove the first row and column of our data
    # To deal with asymmetry in ConvTranpose layers
    # if used correctly, this removes 0's
    def forward(self, x):
        _, _, h, _ = x.shape
        x = x[:, :, 1:h, 1:h]
        return x


# Decode the (Hp x Wp x 8) feature map to it's specific task
class Decoder(nn.Module):
    def __init__(
        self, output_act="none", in_nc=8, out_channels=3, lightweight=True,
    ):
        nn.Module.__init__(self)
        self.output_act = output_act
        act = activations[output_act]()

        if lightweight:
            self.conv2 = self._make_layer(in_nc, 512)
            self.conv3 = self._make_layer(512, 512)
            self.conv4 = nn.Identity()
        else:
            self.conv2 = self._make_layer(8, 1024)
            self.conv3 = self._make_layer(1024, 1024)
            self.conv4 = self._make_layer(1024, 512)

        self.conv5 = self._make_layer(512, 256)
        self.conv6 = self._make_layer(256, 256)
        self.conv7 = self._make_layer(256, 128)

        self.deconv8 = self._make_layer(128, 64, stride=2, deconv=True)
        self.conv9 = self._make_layer(64, 64)

        self.deconv10 = self._make_layer(64, 32, stride=2, deconv=True)
        self.conv11 = self._make_layer(32, 32)

        self.deconv12 = self._make_layer(32, 16, stride=2, deconv=True)
        self.conv13 = self._make_layer(16, 32)

        self.deconv14 = self._make_layer(32, 16, stride=2, deconv=True)
        self.decoder_output = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, bias=True, padding=1),
            act,
        )

    def _make_layer(self, in_channels, out_channels, stride=1, deconv=False):
        if deconv:
            pad = nn.ZeroPad2d((1, 0, 1, 0))  # Pad first row and column
            conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=0,
                bias=False,
            )
            scissor = Scissor()  # Remove first row and column
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )  # pad = 'SAME'

        bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        if deconv:
            layer = nn.Sequential(pad, conv, scissor, bn, lrelu)
        else:
            layer = nn.Sequential(conv, bn, lrelu)
        return layer

    def forward(self, x):
        # Input x: N x 256 x 256 x 3
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.deconv8(x)
        x = self.conv9(x)

        x = self.deconv10(x)
        x = self.conv11(x)

        x = self.deconv12(x)
        x = self.conv13(x)

        x = self.deconv14(x)
        x = self.decoder_output(x)
        # add gaussian-noise?
        return x
