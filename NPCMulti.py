from torch.optim import SGD
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from torchvision.transforms import ToTensor

from models.BA.net.BA import BAT
from util import semantic_to_mask



class DoubleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UAG_RNN(nn.Module):
    """Unidirectional Acyclic Graphs (UCGs)"""

    def __init__(self, in_dim):
        super(UAG_RNN, self).__init__()
        self.chanel_in = in_dim
        self.relu = nn.ReLU()

        self.gamma1 = nn.Parameter(0.5 * torch.ones(1))
        self.gamma2 = nn.Parameter(0.5 * torch.ones(1))
        self.gamma3 = nn.Parameter(0.5 * torch.ones(1))
        self.gamma4 = nn.Parameter(0.5 * torch.ones(1))
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv7 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv8 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv9 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv10 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv11 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv12 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv13 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv14 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv15 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv16 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x, y):
        m_batchsize, C, height, width = x.size()

        ## s plane
        hs = x * 1
        for i in range(height):
            if i > 0:
                hs[:, :, i, :] = self.conv1(hs[:, :, i, :].clone()) + self.conv2(hs[:, :, i - 1, :].clone()) * y[:, :,
                                                                                                               i - 1, :]
                hs[:, :, i, :] = self.relu(hs[:, :, i, :].clone())

        ## s plane
        hse = hs * 1
        for j in range(width):
            if j > 0:
                tmp = self.conv3(hse[:, :, :, j - 1].clone()) * y[:, :, :, j - 1]
                tmp = torch.cat((0 * tmp[:, :, -1].view(m_batchsize, C, 1), tmp[:, :, 0:-1]), 2)  ##diagonal
                hse[:, :, :, j] = self.conv4(hse[:, :, :, j].clone()) + self.conv5(hse[:, :, :, j - 1].clone()) * y[:,
                                                                                                                  :, :,
                                                                                                                  j - 1] + self.gamma1 * tmp
                del tmp
            hse[:, :, :, j] = self.relu(hse[:, :, :, j].clone())

        ## sw plane
        hsw = hs * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                tmp = self.conv6(hsw[:, :, :, j + 1].clone()) * y[:, :, :, j + 1]
                tmp = torch.cat((0 * tmp[:, :, -1].view(m_batchsize, C, 1), tmp[:, :, 0:-1]), 2)  ##diagonal
                hsw[:, :, :, j] = self.conv7(hsw[:, :, :, j].clone()) + self.conv8(hsw[:, :, :, j + 1].clone()) * y[:,
                                                                                                                  :, :,
                                                                                                                  j + 1] + self.gamma2 * tmp
                del tmp
            hsw[:, :, :, j] = self.relu(hsw[:, :, :, j].clone())

        ## n plane
        hn = x * 1
        for i in reversed(range(height)):
            if i < (height - 1):
                hn[:, :, i, :] = self.conv9(hn[:, :, i, :].clone()) + self.conv10(hn[:, :, i + 1, :].clone()) * y[:, :,
                                                                                                                i + 1,
                                                                                                                :]
            hn[:, :, i, :] = self.relu(hn[:, :, i, :].clone())

        ## ne plane
        hne = hn * 1
        for j in range(width):
            if j > 0:
                tmp = self.conv11(hne[:, :, :, j - 1].clone()) * y[:, :, :, j - 1]
                tmp = torch.cat((tmp[:, :, 1:], 0 * tmp[:, :, 0].view(m_batchsize, C, 1)), 2)  ##diagonal
                hne[:, :, :, j] = self.conv12(hne[:, :, :, j].clone()) + self.conv13(hne[:, :, :, j - 1].clone()) * y[:,
                                                                                                                    :,
                                                                                                                    :,
                                                                                                                    j - 1] + self.gamma3 * tmp
                del tmp
            hne[:, :, :, j] = self.relu(hne[:, :, :, j].clone())

        ## nw plane
        hnw = hn * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                tmp = self.conv14(hnw[:, :, :, j + 1].clone()) * y[:, :, :, j + 1]
                tmp = torch.cat((tmp[:, :, 1:], 0 * tmp[:, :, 0].view(m_batchsize, C, 1)), 2)  ##diagonal
                hnw[:, :, :, j] = self.conv15(hnw[:, :, :, j].clone()) + self.conv16(hnw[:, :, :, j + 1].clone()) * y[:,
                                                                                                                    :,
                                                                                                                    :,
                                                                                                                    j + 1] + self.gamma4 * tmp
                del tmp
            hnw[:, :, :, j] = self.relu(hnw[:, :, :, j].clone())

        out = hse + hsw + hnw + hne

        return out

class BFPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(BFPHead, self).__init__()
        inter_channels = in_channels // 4
        self.no_class = out_channels
        self.adapt1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.adapt2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12, padding=12, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.adapt3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12, padding=12, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.uag_rnn = UAG_RNN(inter_channels)
        self.seg1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                  norm_layer(inter_channels),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels + 1, 1))

        self.seg2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                  norm_layer(inter_channels),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1))

        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(2 * torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1) / out_channels)

    def forward(self, x):
        # adapt from CNN
        # print(x.shape)
        feat1 = self.adapt1(x)
        # print(feat1.shape)
        feat2 = self.adapt2(feat1)
        # print(feat1.shape)
        # print(self.seg1)
        # Boundary
        s1_output = self.seg1(feat2)
        # print(s1_output.shape)
        s1_output_ = self.softmax(s1_output)
        # print(s1_output.shape)
        score_ = torch.narrow(s1_output, 1, 0, self.no_class)
        # print(score_.shape)
        boundary_ = torch.narrow(s1_output_, 1, self.no_class, 1)
        # print(boundary_.shape)
        ## boundary confidence to propagation confidence, method 1
        # boundary = 1 - self.sigmoid(20*boundary_-4)*self.gamma

        ## boundary confidence to propagation confidence, method 2
        boundary = torch.mean(torch.mean(boundary_, 2, True), 3, True) - boundary_ + self.bias
        # print(boundary.shape)
        boundary = (boundary - torch.min(torch.min(boundary, 3, True)[0], 2, True)[0]) * self.gamma
        # print(boundary.shape)
        boundary = torch.clamp(boundary, max=1)
        # print(boundary.shape)
        boundary = torch.clamp(boundary, min=0)
        # print(boundary.shape)
        # print('00000')
        ## UAG-RNN
        feat3 = self.adapt3(feat1)
        # print(feat3.shape)
        uag_feat = self.uag_rnn(feat3, boundary)
        # print(uag_feat.shape)
        feat_sum = uag_feat + feat3  # residual
        s2_output = self.seg2(feat_sum)
        # print(s2_output.shape)
        # sd_output = self.conv7(sd_conv)
        output1 = s2_output + score_
        # print(output1.shape)
        output = [output1]  # edge+body
        output.append(s1_output)
        # import pdb
        # pdb.set_trace()
        return tuple(output)

class bfp_newmot1(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(bfp_newmot1, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = DoubleConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = DoubleConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = DoubleConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpBlock(ch_in=1024, ch_out=512)
        self.Up_conv5 = DoubleConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpBlock(ch_in=512, ch_out=256)
        self.Up_conv4 = DoubleConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpBlock(ch_in=256, ch_out=128)
        self.Up_conv3 = DoubleConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpBlock(ch_in=128, ch_out=64)
        self.Up_conv2 = DoubleConvBlock(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.head = BFPHead(1024, 1024, nn.BatchNorm2d)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.head(x5)
        # print(x5[0].shape,x5[1].shape)
        # print(x5.shape)
        x5 = x5[0]
        # decoding + concat path
        d5 = self.Up5(x5)

        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # x2 = x2.to('cpu')
        # d3 = d3.to('cpu')
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d1 = d1.to(self.device)
        return x1,x2,x3,x4,x5,d1

class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = DoubleConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = DoubleConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = DoubleConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpBlock(ch_in=1024, ch_out=512)
        self.Up_conv5 = DoubleConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpBlock(ch_in=512, ch_out=256)
        self.Up_conv4 = DoubleConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpBlock(ch_in=256, ch_out=128)
        self.Up_conv3 = DoubleConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpBlock(ch_in=128, ch_out=64)
        self.Up_conv2 = DoubleConvBlock(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

class DilatedEncoder3(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,in_channels,num_channels,mid_channels,out_channels,num_residual_blocks,dilations,norm=nn.BatchNorm2d,activation=nn.ReLU):
        super(DilatedEncoder3, self).__init__()
        # fmt: off
        self.in_channels = in_channels
        self.encoder_channels = num_channels
        self.block_mid_channels = mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = dilations
        self.norm_type = norm
        self.act_type = activation
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)


        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = self.norm_type(self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = self.norm_type(self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck3(
                    self.encoder_channels + self.out_channels*i,
                    self.out_channels,
                    # self.in_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # print(' ')
        # print(feature.shape)
        # out = self.lateral_norm(self.lateral_conv(feature))
        # out = self.fpn_norm(self.fpn_conv(out))
        # print(out.shape)
        # return self.dilated_encoder_blocks(out)
        x = self.conv1(feature)
        return self.dilated_encoder_blocks(x)

class Bottleneck3(nn.Module):

    def __init__(self,
                 in_channels = 512,
                 out_channels=1,
                 mid_channels = 128,
                 dilation = 1,
                 norm_type = nn.BatchNorm2d,
                 act_type = nn.ReLU):
        super(Bottleneck3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            # norm_type(mid_channels),
            # act_type(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            # norm_type(mid_channels),
            # act_type(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,kernel_size=5, padding=dilation*2, dilation=dilation),
            # norm_type(mid_channels),
            # act_type(act_type)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            # norm_type(mid_channels),
            # act_type(act_type)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
            # norm_type(in_channels),
            # act_type(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # out = out + identity

        out = torch.cat([out,identity],dim=1)
        return out

class newmot10_gpu(nn.Module):
    def __init__(self,interval,device, img_ch=3, output_ch=3):
        super(newmot10_gpu, self).__init__()
        self.device = device
        self.UNet = bfp_newmot1()
        self.bat = BAT(1,1312)
        # self.conv1 = nn.Conv2d(1024,128, kernel_size=3, stride=2, padding=1, bias=True).to(device)
        self.conv1 = nn.Conv2d(1312,128, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256,interval)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(73732, 1024)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConvBlock(ch_in=70, ch_out=128)
        self.Conv3 = DoubleConvBlock(ch_in=140, ch_out=256)
        self.Conv4 = DoubleConvBlock(ch_in=296, ch_out=512)
        self.Conv5 = DoubleConvBlock(ch_in=592, ch_out=1024)

        self.seg_Conv1 = DilatedEncoder3(in_channels=64, num_channels=4, mid_channels=32,out_channels=1, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv2 = DilatedEncoder3(in_channels=128, num_channels=8, mid_channels=64,out_channels=2, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv3 = DilatedEncoder3(in_channels=256, num_channels=32, mid_channels=128,out_channels=4, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv4 = DilatedEncoder3(in_channels=512, num_channels=64, mid_channels=256,out_channels=8, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv5 = DilatedEncoder3(in_channels=1024, num_channels=256, mid_channels=512,out_channels=16, num_residual_blocks=2,
                       dilations=[3,1])
    # def runUNet(self,x):
    #     y = self.UNet(x)
    #     return y
    # def seg(self,seg_x1,seg_x2,seg_x3,seg_x4,seg_x5):
    #     seg_x1 = self.seg_Conv1(seg_x1)
    #     seg_x2 = self.seg_Conv2(seg_x2)
    #     seg_x3 = self.seg_Conv3(seg_x3)
    #     seg_x4 = self.seg_Conv4(seg_x4)
    #     seg_x5 = self.seg_Conv5(seg_x5)
    #     return seg_x1,seg_x2,seg_x3,seg_x4,seg_x5
    def run_model(self, x,clinical_data):


        # print(x)


        seg_x1,seg_x2,seg_x3,seg_x4, seg_x5,d1 = self.UNet(x)
        # seg_x1 = seg_x1.to('cpu')
        # seg_x2 = seg_x2.to('cpu')
        # seg_x3 = seg_x3.to('cpu')
        # seg_x4 = seg_x4.to('cpu')
        # seg_x5 = seg_x5.to('cpu')

        seg_x1 = self.seg_Conv1(seg_x1)
        seg_x2 = self.seg_Conv2(seg_x2)
        seg_x3 = self.seg_Conv3(seg_x3)
        seg_x4 = self.seg_Conv4(seg_x4)
        seg_x5 = self.seg_Conv5(seg_x5)

        d2 = d1.cpu().detach().numpy()
        mask = semantic_to_mask(d2, [0,1,2])
        mask = np.where(mask>=1,1,0)
        mask1 = np.zeros([x.shape[0],3,256,256])
        for i in range(x.shape[0]):
            mask1[i, 0, :, :] = mask[i]
            mask1[i, 1, :, :] = mask[i]
            mask1[i, 2, :, :] = mask[i]


        xx = x.cpu().detach().numpy()
        y = np.multiply(xx,mask1)
        y = y.astype(np.float32)
        # x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y)
        # x = x.to('cpu')
        # x = torch.cat([x,y],dim = 1)
        y = y.to(self.device)
        x1 = self.Conv1(y)
        x1 = torch.cat([x1,seg_x1],dim=1)
        # x1 = x1.to('cpu')
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = torch.cat([x2,seg_x2],dim=1)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = torch.cat([x3,seg_x3],dim=1)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = torch.cat([x4,seg_x4],dim=1)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = torch.cat([x5,seg_x5],dim=1)
        # x5 = torch.cat([x5, seg_feature], dim=1)
        # x5 = x5.to(self.device)
        x5 = self.bat(x5)
        #
        #
        #
        # seg5 = x5.to(self.device1)
        # seg = self.recover(seg5)

        sur = self.conv1(x5)

        sur = self.Maxpool(sur)

        sur = torch.flatten(sur, 0)
        # clinical_data = clinical_data.to('cpu')
        sur = torch.cat((clinical_data, sur), dim=0)

        if sur.shape[0]!= 73732:
            patches = torch.zeros(size=(73732-sur.shape[0],),dtype=sur.dtype).to(self.device)
            # patches = torch.zeros(size=(73732 - sur.shape[0],), dtype=sur.dtype)
            sur = torch.cat((sur,patches),dim=0)

        sur = self.fc2(sur)

        sur = self.relu2(sur)
        # sur = sur.to(self.device)
        sur = self.fc3(sur)
        # sur = self.relu3(sur)
        sur = self.fc4(sur)
        sur = self.sigmoid(sur)
        # sur = sur.to(self.device)
        return d1, sur
    def forward(self, x,clinical_data):
        x = x+torch.zeros(1,dtype=x.dtype,device=x.device,requires_grad=True)
        clinical_data = clinical_data + torch.zeros(1, dtype=clinical_data.dtype, device=clinical_data.device, requires_grad=True)
        d1,sur = checkpoint(self.run_model,x,clinical_data)
        return d1,sur

class NPCMulti(nn.Module):
    def __init__(self,interval,device, img_ch=3, output_ch=3):
        super(NPCMulti, self).__init__()
        self.device = device

        self.UNet = bfp_newmot1()
        self.bat = BAT(1,1312)
        # self.conv1 = nn.Conv2d(1024,128, kernel_size=3, stride=2, padding=1, bias=True).to(device)
        self.conv1 = nn.Conv2d(1312,128, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu2 = nn.ReLU().to(device)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256,interval)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(73732, 1024)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConvBlock(ch_in=70, ch_out=128)
        self.Conv3 = DoubleConvBlock(ch_in=140, ch_out=256)
        self.Conv4 = DoubleConvBlock(ch_in=296, ch_out=512)
        self.Conv5 = DoubleConvBlock(ch_in=592, ch_out=1024)

        self.seg_Conv1 = DilatedEncoder3(in_channels=64, num_channels=4, mid_channels=32,out_channels=1, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv2 = DilatedEncoder3(in_channels=128, num_channels=8, mid_channels=64,out_channels=2, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv3 = DilatedEncoder3(in_channels=256, num_channels=32, mid_channels=128,out_channels=4, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv4 = DilatedEncoder3(in_channels=512, num_channels=64, mid_channels=256,out_channels=8, num_residual_blocks=2,
                       dilations=[3,1])
        self.seg_Conv5 = DilatedEncoder3(in_channels=1024, num_channels=256, mid_channels=512,out_channels=16, num_residual_blocks=2,
                       dilations=[3,1])
    def forward(self, x):


        # print(x)


        seg_x1,seg_x2,seg_x3,seg_x4, seg_x5,d1 = self.UNet(x)

        seg_x1 = self.seg_Conv1(seg_x1)
        seg_x2 = self.seg_Conv2(seg_x2)
        seg_x3 = self.seg_Conv3(seg_x3)
        seg_x4 = self.seg_Conv4(seg_x4)
        seg_x5 = self.seg_Conv5(seg_x5)

        d2 = d1.cpu().detach().numpy()
        mask = semantic_to_mask(d2, [0,1,2])
        mask = np.where(mask>=1,1,0)
        mask1 = np.zeros([x.shape[0],3,256,256])
        for i in range(x.shape[0]):
            mask1[i, 0, :, :] = mask[i]
            mask1[i, 1, :, :] = mask[i]
            mask1[i, 2, :, :] = mask[i]


        xx = x.cpu().detach().numpy()
        y = np.multiply(xx,mask1)
        y = y.astype(np.float32)
        # x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y)
        # x = x.to('cpu')
        # x = torch.cat([x,y],dim = 1)
        y = y.to(self.device)
        x1 = self.Conv1(y)
        x1 = torch.cat([x1,seg_x1],dim=1)
        # x1 = x1.to('cpu')
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = torch.cat([x2,seg_x2],dim=1)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = torch.cat([x3,seg_x3],dim=1)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = torch.cat([x4,seg_x4],dim=1)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = torch.cat([x5,seg_x5],dim=1)
        # x5 = torch.cat([x5, seg_feature], dim=1)
        x5 = x5.to(self.device)
        x5 = self.bat(x5)
        #
        #
        #
        # seg5 = x5.to(self.device1)
        # seg = self.recover(seg5)

        sur = self.conv1(x5)

        sur = self.Maxpool(sur)

        sur = torch.flatten(sur, 0)


        if sur.shape[0]!= 73728:
            patches = torch.zeros(size=(73728-sur.shape[0],),dtype=sur.dtype).to(self.device)

            sur = torch.cat((sur,patches),dim=0)

        sur = self.fc2(sur)

        sur = self.relu2(sur)
        # sur = sur.to(self.device)
        sur = self.fc3(sur)
        # sur = self.relu3(sur)
        sur = self.fc4(sur)
        sur = self.sigmoid(sur)
        # sur = sur.to(self.device)
        return d1, sur


if __name__ == '__main__':
    k = 2
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    clinical_data = torch.rand(4, ).to(device)
    net = NPCMulti(18,device)
    net = net.to(device)

    img = torch.rand(k, 3, 256, 256,dtype=torch.float32).to(device)
    x = net(img)
    # x = net(img,clinical_data)
    print(x[1].shape)
