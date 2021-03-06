import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd

# Basic utilities
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

# Implementation from mmcv.cnn.utils.weight_init
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# Implementation from mmcv.cnn.utils.weight_init
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        # nn.init.constant_(m[-1], 0)  # Changed mmcv to pytorch
        # m[-1].inited = True
    else:
        constant_init(m, val=0)
        # nn.init.constant_(m, 0) # Changed mmcv to pytorch
        m.inited = True

# Squeeze-and-excitation networks, CVPR 2017
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class se_layer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(se_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.c2(self.c1(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


# CBAM: Convolutional Block Attention Module, ECCV 2018
# https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # Adaptiveavgpool?
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # Adaptivemaxpool?
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class cbam(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(cbam, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, ICCV 2019
# https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
class ContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            # nn.init.kaiming_uniform_(self.conv_mask, nonlinearity='relu')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        # if pooling tupe is 'att', it performs as the simple NL block
        # else, just simple global average pooling
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


# Dual Channel Attention Networks, Journal of Physics 2020
# https://github.com/13952522076/DCANet/blob/master/models/resnet/resnet_gc.py
class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add']):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out



# ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, CVPR 2020
class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class simple_feat_layer(nn.Module):
    def __init__(self, num_channel, layer_num, debug=False):
        super(simple_feat_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(num_channel, num_channel, bias=False)
        self.fc2 = nn.Linear(num_channel, num_channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.layer_num = layer_num
        self.debug = debug
        self.relu = nn.ReLU()
        # self.patch_num = patch_num

    def forward(self, x):
        b, c, h, w = x.size()
        # print(b, c, h, w)
        y = self.avg_pool(x).view(b, c)
        # y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))

        if self.debug:
            # print(y)
            y_np = y.to('cpu').numpy()
            y_df = pd.DataFrame(y_np)
            y_df.to_csv('acts/Act_{}.csv'.format(self.layer_num))
            
        y = y.view(b, c, 1, 1)
        # print(y.shape)
        return x * y.expand((b, c, h, w))   # 0.5??? ????????????
        # return x * 0.5


# spatial-pyramid channel attention module
class spc_layer(nn.Module):
    def __init__(self, num_channel, layer_num, level=2, debug=False):
        super(spc_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.fc_list = nn.ModuleList()
        n = 0
        for i in range(level+1):
            n += 4 ** i

        # print(n)/
        for j  in range(n):
            self.fc_list.append(nn.Linear(num_channel, num_channel, bias=False))

        self.layer_num = layer_num
        print(self.layer_num)
        self.debug = debug

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            b, c, h, w = x.size()
            # Level 0
            avg_0 = self.avg_pool(x).view(b, c)
            out_avg_0 = self.fc_list[0](avg_0)
            max_0 = self.max_pool(x).view(b, c)
            out_max_0 = self.fc_list[0](max_0)
            out_0 = self.sigmoid(out_avg_0 + out_max_0)
            if self.debug:
                out_0_np = out_0.to('cpu').numpy()
                out_0_df = pd.DataFrame(out_0_np)
                out_0_df.to_csv('acts/spc/Act_{}_0.csv'.format(self.layer_num))
            out_0 = out_0.view(b, c, 1, 1)
            x_0 = x * out_0.expand((b, c, h, w))


            # Level 1
            # out_1_list = []
            # out_1 = None
            x_1 = x.clone()
            # print(x.shape)
            for i in range(2):
                for j in range(2):
                    x_sub_1 = x[:, :, i*(h//2):(i+1) * (h//2), j*(w//2):(j+1)*(w//2)]
                    # print(x_sub_1.shape)
                    avg_1 = self.avg_pool(x[:, :, i*(h//2):(i+1) * (h//2), j*(w//2):(j+1)*(w//2)])
                    # print(avg_1.shape)
                    out_avg_1 = self.fc_list[1+2*i+j](avg_1.squeeze(-1).squeeze(-1))
                    max_1 = self.max_pool(x[:, :, i*(h//2):(i+1) * (h//2), j*(w//2):(j+1)*(w//2)])
                    out_max_1 = self.fc_list[1+2*i+j](max_1.squeeze(-1).squeeze(-1))
                    # out_1_list.append(self.sigmoid(out_avg_1 + out_max_1))
                    out_1 = self.sigmoid(out_avg_1 + out_max_1)
                    # print(out_1.shape)
                    # out_1_list.append(x_sub_1 * out_1.expand_as(x_sub_1))

                    if self.debug:
                        out_1_np = out_1.to('cpu').numpy()
                        out_1_df = pd.DataFrame(out_1_np)
                        out_1_df.to_csv('acts/spc/Act_{}_{}.csv'.format(self.layer_num, 2*i+j + 1))

                    out_1 = out_1.view(b, c, 1, 1)
                    x_1[:, :, i*(h//2):(i+1) * (h//2), j*(w//2):(j+1)*(w//2)] = x_sub_1 * out_1.expand_as(x_sub_1)


            # Level 2
            # out_2_list = []
            x_2 = x.clone()
            for i in range(4):
                for j in range(4):
                    x_sub_2 = x[:, :, i*(h//4):(i+1) * (h//4), j*(w//4):(j+1)*(w//4)]
                    avg_2 = self.avg_pool(x[:, :, i*(h//4):(i+1) * (h//4), j*(w//4):(j+1)*(w//4)])
                    out_avg_2 = self.fc_list[5+4*i+j](avg_2.squeeze(-1).squeeze(-1))
                    max_2 = self.max_pool(x[:, :, i*(h//4):(i+1) * (h//4), j*(w//4):(j+1)*(w//4)])
                    out_max_2 = self.fc_list[5+4*i+j](max_2.squeeze(-1).squeeze(-1))
                    # out_2_list.append(self.sigmoid(out_avg_2 + out_max_2))
                    out_2 = self.sigmoid(out_avg_2 + out_max_2)
                    # out_2_list.append(x_sub_2 * out_2.expand_as(x_sub_2))

                    if self.debug:
                        out_2_np = out_2.to('cpu').numpy()
                        out_2_df = pd.DataFrame(out_2_np)
                        out_2_df.to_csv('acts/spc/Act_{}_{}.csv'.format(self.layer_num, 4*i+j + 5))

                    out_2 = out_2.view(b, c, 1, 1)
                    x_2[:, :, i*(h//4):(i+1) * (h//4), j*(w//4):(j+1)*(w//4)] = x_sub_2 * out_2.expand_as(x_sub_2)

            # print(x_0.shape, x_1.shape, x_2.shape)
            return x_0

# psc_layer(34)


class nlblock(nn.Module):
    def __init__(self, input_channels, inter_channels):
        super(nlblock, self).__init__()
        self.inter_channels = inter_channels
        self.theta = nn.Conv2d(input_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(input_channels, inter_channels, kernel_size=1)
        self.g = nn.Conv2d(input_channels, inter_channels, kernel_size=1)
        
        self.W = nn.Conv2d(inter_channels, input_channels, kernel_size=1)
        

    def forward(self, x):
        # x = B * C * H * W
        # print(x.shape)
        B, C, H, W = x.shape
        key = self.theta(x).view(B, self.inter_channels, -1)
        value = self.phi(x).view(B, self.inter_channels, -1)
        query = self.g(x).view(B, self.inter_channels, -1)
        # key = x.view(B, self.inter_channels, -1)
        # value = x.view(B, self.inter_channels, -1)
        # query = x.view(B, self.inter_channels, -1)
        # print(key.shape)
        key = key.permute(0, 2, 1)
        query = query.permute(0, 2, 1)

        score = torch.matmul(key, value)
        # print(score.shape)
        # print(score)
        score = F.softmax(score, dim=-1) # B * C * HW * HW
        # score = nn.Softmax(dim=-1)(score)
        # print(score)
        scaled = torch.matmul(score, query)
        # scaled = scaled.permute(0, 2, 1).contiguous()
        del score
        del query
        del key
        del value
        scaled = scaled.permute(0, 2, 1)
        scaled = scaled.view(B, self.inter_channels, H, W)
        # print(scaled)
        output = self.W(scaled)
        del scaled
        # output = scaled
        return output+x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
        
class sablock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(sablock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        x_res = self.relu(x_res)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        y  = self.sigmoid(y)
        # print(y.shape, x_res.shape)
        # return (y * x_res) + y
        return x * y


# nlblock = nlblock(2, 2)
# x = torch.rand((2, 2, 3, 3))
# print(x)
# y = nlblock(x)
# print(y.shap[e])
# print(y)
