'''
2022.9.30
prostate数据集使用的unet
'''
import sys
sys.path.append(sys.path[0].replace('networks', ''))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.amp_utils import process

"""
Wrappers for the operations to take the meta-learning gradient
updates into account.
"""
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)

def conv2d(inputs, weight, bias, stride=1, padding=1, dilation=1, groups=1, kernel_size=3):

    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def deconv2d(inputs, weight, bias, stride=2, padding=0, dilation=0, groups=1, kernel_size=None):

    inputs = inputs.cuda()
    weight = weight.cuda()
    bias = bias.cuda()

    return F.conv_transpose2d(inputs, weight, bias, stride, padding, dilation, groups)

def relu(inputs):
    return F.relu(inputs, inplace=True)


def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)


def dropout(inputs):
    return F.dropout(inputs, p=0.5, training=False, inplace=False)

def batchnorm(inputs, running_mean, running_var):
    return F.batch_norm(inputs, running_mean, running_var)


"""
The following are the new methods for 2D-Unet:
Conv2d, batchnorm2d, GroupNorm, InstanceNorm2d, MaxPool2d, UpSample
"""
#as per the 2D Unet:  kernel_size, stride, padding

def instancenorm(input):
    return F.instance_norm(input)

def groupnorm(input):
    return F.group_norm(input)

def dropout2D(inputs):
    return F.dropout2d(inputs, p=0.5, training=False, inplace=False)

def maxpool2D(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)

def upsample(input):
    return F.upsample(input, scale_factor=2, mode='bilinear', align_corners=False)


class AmpNorm(nn.Module):
    def __init__(self, input_shape, momentum=0.1):
        super(AmpNorm, self).__init__()
        self.register_buffer('running_amp', torch.zeros(input_shape))
        self.momentum = momentum
        self.fix_amp = False     
        
    def forward(self, x):
        device = x.device
        if not self.fix_amp:
            if torch.sum(self.running_amp) == 0:
                x, amp = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)
                self.running_amp = torch.from_numpy(amp)
            else:
                x, amp = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)
                self.running_amp = torch.from_numpy(amp)
        else:
            x, _ = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)

        return torch.from_numpy(x).to(device)
    
class UNet(nn.Module):
    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32, norm_type='no'):
        super(UNet, self).__init__()
        self.norm_type = norm_type
        self.amp_norm = AmpNorm(input_shape=input_shape)

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", norm_type=self.norm_type)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", norm_type=self.norm_type)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", norm_type=self.norm_type)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", norm_type=self.norm_type)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", norm_type=self.norm_type)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", norm_type=self.norm_type)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", norm_type=self.norm_type)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1", norm_type=self.norm_type)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, feature_out=False):
        x = self.amp_norm(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        pred = self.conv(dec1)
        if feature_out:
            return pred, dec1
        return pred

    @staticmethod
    def _block(in_channels, features, name, norm_type='no'):
        if norm_type == 'no':
            my_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                ]
            )
        )
        elif norm_type == 'bn':
            my_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=True, track_running_stats=True)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=True, track_running_stats=True)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        elif norm_type == 'in':
            my_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_in1", nn.InstanceNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_in2", nn.InstanceNorm2d(num_features=features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
        else:
            raise NotImplementedError
        
        return my_block
        

class UNet_ori(nn.Module):
    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32):
        super(UNet_ori, self).__init__()

        self.amp_norm = AmpNorm(input_shape=input_shape)

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1",)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, feature_out=False):
        x = self.amp_norm(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        pred = self.conv(dec1)
        if feature_out:
            return pred, dec1
        return pred

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
