# Implementation of Deep Convolutional Reconstruction For Gradient-Domain Rendering
# SIGGRAPH 2019
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from basic_models import simple_feat_layer


class ConvUnit(nn.Module):
    def __init__(self, in_channels, N, mid_channels=None):
        super(ConvUnit, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels, 2*N, kernel_size=1, padding=0)
        self.Conv2 = nn.Conv2d(2*N, N, kernel_size=3, padding=1)
        self.LReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.LReLU(self.Conv1(x))
        return self.LReLU(self.Conv2(out))


class ResolUnit4(nn.Module):
    def __init__(self, in_channels, N, channel_attn=False):
        super(ResolUnit4, self).__init__()
        self.Conv1 = ConvUnit(in_channels, N)
        self.Conv2 = ConvUnit(in_channels+N, N)
        self.Conv3 = ConvUnit(in_channels+2*N, N)
        self.Conv4 = ConvUnit(in_channels+3*N, N)
        self.channel_attn = None
        if channel_attn:
            self.feat_attn = simple_feat_layer(in_channels+4*N)
    
    def forward(self, x):
        out1 = torch.cat((self.Conv1(x), x), dim=1)
        out2 = torch.cat((self.Conv2(out1), out1), dim=1)
        out3 = torch.cat((self.Conv3(out2), out2), dim=1)
        out4 = torch.cat((self.Conv4(out3), out3), dim=1)
        if self.channel_attn:
            return self.feat_attn(out4)
        return out4

class ResolUnit3(nn.Module):
    def __init__(self, in_channels, N, channel_attn=False):
        super(ResolUnit3, self).__init__()
        self.Conv1 = ConvUnit(in_channels, N)
        self.Conv2 = ConvUnit(in_channels+N, N)
        self.Conv3 = ConvUnit(in_channels+2*N, N)
        self.channel_attn = None
        if channel_attn:
            self.feat_attn = simple_feat_layer(in_channels+3*N)
    
    def forward(self, x):
        out1 = torch.cat((self.Conv1(x), x), dim=1)
        out2 = torch.cat((self.Conv2(out1), out1), dim=1)
        out3 = torch.cat((self.Conv3(out2), out2), dim=1)
        if self.channel_attn:
            return self.feat_attn(out3)
        return out3


class ResolUnit2(nn.Module):
    def __init__(self, in_channels, N, channel_attn=False):
        super(ResolUnit2, self).__init__()
        self.Conv1 = ConvUnit(in_channels, N)
        self.Conv2 = ConvUnit(in_channels+N, N)
        self.channel_attn = None
        if channel_attn:
            self.feat_attn = simple_feat_layer(in_channels+2*N)
    
    def forward(self, x):
        out1 = torch.cat((self.Conv1(x), x), dim=1)
        out2 = torch.cat((self.Conv2(out1), out1), dim=1)
        if self.channel_attn:
            return self.feat_attn(out2)
        return out2


class DenseUnet(nn.Module):
    def __init__(self, input_channels, mode, channel_attn=False):
        super(DenseUnet, self).__init__()
        self.LReLU = nn.LeakyReLU(inplace=True)

        # Encoder part
        self.ResolUnit4_1 = ResolUnit4(input_channels, 40, channel_attn)
        self.ResolUnit3_1 = ResolUnit3(160, 80, channel_attn)
        self.ResolUnit2_1 = ResolUnit2(160, 80, channel_attn)
        self.ResolUnit4_2 = ResolUnit4(160, 80, channel_attn)
        self.down_1 = nn.Conv2d(input_channels+160, 160, kernel_size=2, stride=2)
        self.down_2 = nn.Conv2d(400, 160, kernel_size=2, stride=2)
        self.down_3 = nn.Conv2d(320, 160, kernel_size=2, stride=2)

        # Decoder part
        self.ResolUnit2_2 = ResolUnit2(480, 80, channel_attn)
        self.ResolUnit3_2 = ResolUnit3(560, 80, channel_attn)
        self.ResolUnit4_3 = ResolUnit4(input_channels+240, 40, channel_attn)
        self.up_1 = nn.ConvTranspose2d(480, 160, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(640, 160, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(800, 80, kernel_size=2, stride=2)

        # filtering
        if 'kpcn' in mode:
            self.filter = nn.Conv2d(input_channels+400, 441, kernel_size=5)
        else:
            self.filter = nn.Conv2d(input_channels+400, 3, kernel_size=5)

        
    def forward(self, x):
        # X = B * C * H * W
        enc1 = self.ResolUnit4_1(x)
        down1 = self.down_1(enc1)
        enc2 = self.ResolUnit3_1(down1)
        down2 = self.down_2(enc2)
        enc3 = self.ResolUnit2_1(down2)
        down3 = self.down_3(enc3)
        enc4 = self.ResolUnit4_2(down3)

        up1 = self.up_1(enc4)
        dec1 = self.ResolUnit2_2(torch.cat((up1, enc3), dim=1))
        up2 = self.up_2(dec1)
        dec2 = self.ResolUnit3_2(torch.cat((up2, enc2), dim=1))
        up3 = self.up_3(dec2)
        dec3 = self.ResolUnit4_3(torch.cat((up3, enc1), dim=1))

        return self.filter(dec3)


class simplefeatUnetKPCN(nn.Module):
    def __init__(self, input_channels):
        super(simplefeatUnetKPCN, self).__init__()
        self.LReLU = nn.LeakyReLU(inplace=True)

        # Encoder part
        self.ResolUnit4_1 = ResolUnit4(input_channels, 40)
        self.ResolUnit3_1 = ResolUnit3(160, 80)
        self.ResolUnit2_1 = ResolUnit2(160, 80)
        self.ResolUnit4_2 = ResolUnit4(160, 80)
        self.down_1 = nn.Conv2d(input_channels+160, 160, kernel_size=2, stride=2)
        self.down_2 = nn.Conv2d(400, 160, kernel_size=2, stride=2)
        self.down_3 = nn.Conv2d(320, 160, kernel_size=2, stride=2)

        # Decoder part
        self.ResolUnit2_2 = ResolUnit2(480, 80)
        self.ResolUnit3_2 = ResolUnit3(560, 80)
        self.ResolUnit4_3 = ResolUnit4(input_channels+240, 40)
        self.up_1 = nn.ConvTranspose2d(480, 160, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(640, 160, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(800, 80, kernel_size=2, stride=2)

        self.filter = nn.Conv2d(input_channels+400, 441, kernel_size=5)
        
    def forward(self, x):
        # X = B * C * H * W
        enc1 = self.ResolUnit4_1(x)
        down1 = self.down_1(enc1)
        enc2 = self.ResolUnit3_1(down1)
        down2 = self.down_2(enc2)
        enc3 = self.ResolUnit2_1(down2)
        down3 = self.down_3(enc3)
        enc4 = self.ResolUnit4_2(down3)

        up1 = self.up_1(enc4)
        dec1 = self.ResolUnit2_2(torch.cat((up1, enc3), dim=1))
        up2 = self.up_2(dec1)
        dec2 = self.ResolUnit3_2(torch.cat((up2, enc2), dim=1))
        up3 = self.up_3(dec2)
        dec3 = self.ResolUnit4_3(torch.cat((up3, enc1), dim=1))

        return self.filter(dec3)


# Implementation of Learning to reconstruct shape and spatially-varying reflectance from a single image
# SiGGRAPH ASIA 2018
# https://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF/
class encoderUnet(nn.Module):
    def __init__(self, in_channel=3):
        super(encoderUnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.bn1 = nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        self.bn2 = nn.Identity()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)
        self.bn3 = nn.Identity()
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        # # self.bn4 = nn.BatchNorm2d(256)
        # self.bn4 = nn.Identity()
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        # # self.bn5 = nn.BatchNorm2d(512)
        # self.bn5 = nn.Identity()
        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        # print('INPUT : {}'.format(x.shape))
        x1 = F.relu(self.bn1(self.conv1(x)), True) # C=32
        x2 = F.relu(self.bn2(self.conv2(x1)), True) # C=64
        x3 = F.relu(self.bn3(self.conv3(x2)), True) # C=128
        # x4 = F.relu(self.bn4(self.conv4(x3)), True) # C=256
        # x5 = F.relu(self.bn5(self.conv5(x4)), True) # C=512
        # x = F.relu(self.bn6(self.conv6(x5)), True) # C=512
        # return x1, x2, x3, x4, x5
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # return x1, x2, x3, x4
        return x1, x2, x3

class decoderUnet(nn.Module):
    def __init__(self, out_channel=3, mode=0):
        super(decoderUnet, self).__init__()
        # self.conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(512)
        # self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn2 = nn.Identity()
        # self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False) # in_channels 256+256
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn3 = nn.Identity()
        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False) # in_channels 128+128
        # self.bn4 = nn.BatchNorm2d(64)
        self.bn4 = nn.Identity()
        self.conv5 = nn.ConvTranspose2d(in_channels=64+64, out_channels=32, kernel_size=4, stride=2, padding=2, bias=False)
        # self.bn5 = nn.BatchNorm2d(32)
        self.bn5 = nn.Identity()
        self.conv6 = nn.ConvTranspose2d(in_channels=32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(64)
        self.bn6 = nn.Identity()
        self.conv7 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.mode = mode

        self.sigmoid = nn.Sigmoid()

    # def forward(self, x1, x2, x3, x4, x5, x):
    #     y1 = F.relu(self.bn1(self.conv1(x)), True) # C=512
    #     y2 = F.relu(self.bn2(self.conv2(torch.cat((y1, x5), dim=1))), True) # C=256
    #     y3 = F.relu(self.bn3(self.conv3(torch.cat((y2, x4), dim=1))), True) # C=256
    #     y4 = F.relu(self.bn4(self.conv4(torch.cat((y3, x3), dim=1))), True) # C=128
    #     y5 = F.relu(self.bn5(self.conv5(torch.cat((y4, x2), dim=1))), True) # C=64
    #     y6 = F.relu(self.bn6(self.conv6(torch.cat((y5, x1), dim=1))), True) # C=32
    #     y = self.conv7(y6)
    #     return y

    # def forward(self, x1, x2, x3, x4, x5):
    #     # print(x4.shape)
    #     y1 = F.relu(self.bn2(self.conv2(x5)), True) # 256
    #     # print(y1.shape, x3.shape)
    #     y2 = F.relu(self.bn3(self.conv3(torch.cat((y1, x4), dim=1))), True) # 256
    #     y3 = F.relu(self.bn4(self.conv4(torch.cat((y2, x3), dim=1))), True)
    #     y4 = F.relu(self.bn5(self.conv5(torch.cat((y3, x2), dim=1))), True)
    #     y5 = F.relu(self.bn6(self.conv6(torch.cat((y4, x1), dim=1))), True)
    #     y = self.conv7(y5)
    #     # yout = y

    #     if self.mode == 0: # albedo
    #         # yout = torch.clamp(y, min=0.0, max=1.0) # 5
    #         yout = self.sigmoid(y) # 4
    #     elif self.mode == 1: # normal
    #         norm = torch.sqrt(torch.sum(y * y, dim=1).unsqueeze(1) ).expand_as(y)
    #         yout = y / norm
    #     elif self.mode == 2: # depth
    #         yout = torch.clamp(y, min=-0.1, max=1.0)
    #     return yout

    # def forward(self, x1, x2, x3, x4):
    #     # print(x4.shape)
    #     y2 = F.relu(self.bn3(self.conv3(x4)), True) # 256
    #     # print(y1.shape, x3.shape)
    #     # y2 = F.relu(self.bn3(self.conv3(torch.cat((y1, x4), dim=1))), True) # 256
    #     y3 = F.relu(self.bn4(self.conv4(torch.cat((y2, x3), dim=1))), True)
    #     y4 = F.relu(self.bn5(self.conv5(torch.cat((y3, x2), dim=1))), True)
    #     y5 = F.relu(self.bn6(self.conv6(torch.cat((y4, x1), dim=1))), True)
    #     y = self.conv7(y5)
    #     # yout = y

    #     if self.mode == 0: # albedo
    #         # yout = torch.clamp(y, min=0.0, max=1.0) # 5
    #         yout = self.sigmoid(y) # 4
    #     elif self.mode == 1: # normal
    #         norm = torch.sqrt(torch.sum(y * y, dim=1).unsqueeze(1) ).expand_as(y)
    #         yout = y / norm
    #     elif self.mode == 2: # depth
    #         yout = torch.clamp(y, min=-0.1, max=1.0)
    #     return yout

    def forward(self, x1, x2, x3):
        # print(x4.shape)
        # y2 = F.relu(self.bn3(self.conv3(x4)), True) # 256
        # print(y1.shape, x3.shape)
        # y2 = F.relu(self.bn3(self.conv3(torch.cat((y1, x4), dim=1))), True) # 256
        # y3 = F.relu(self.bn4(self.conv4(torch.cat((y2, x3), dim=1))), True)
        y3 = F.relu(self.bn4(self.conv4(x3)), True)
        y4 = F.relu(self.bn5(self.conv5(torch.cat((y3, x2), dim=1))), True)
        y5 = F.relu(self.bn6(self.conv6(torch.cat((y4, x1), dim=1))), True)
        y = self.conv7(y5)
        # yout = y

        if self.mode == 0: # albedo
            # yout = torch.clamp(y, min=0.0, max=1.0) # 5
            yout = self.sigmoid(y) # 4
        elif self.mode == 1: # normal
            norm = torch.sqrt(torch.sum(y * y, dim=1).unsqueeze(1) ).expand_as(y)
            yout = y / norm
        elif self.mode == 2: # depth
            yout = torch.clamp(y, min=-0.1, max=1.0)
        return yout


class sampleUnet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(sampleUnet, self).__init__()
        self.encoder = encoderUnet(in_channel=in_channel)
        self.decoder = decoderUnet(out_channel=out_channel)

    def forward(self, x):
        y1, y2, y3, y4, y5, y = self.encoder(x)
        return self.decoder(y1, y2, y3, y4, y5, y)


# TESTING


unet = sampleUnet(in_channel=1, out_channel=1)
t = torch.ones((8, 1, 96, 96))
print(t.shape)
print((t).shape)