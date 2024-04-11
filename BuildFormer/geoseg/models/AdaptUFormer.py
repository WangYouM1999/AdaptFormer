import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNGELU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNGELU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU()
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class DWLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

        # (23,1) -> （5，1） + （7，3）
        self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)

        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.proj_2(x)
        x = x + shorcut

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwlk = DWLK(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwlk(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeatureAdaptMixer(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)

        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # adaptive spatial fusion
        self.hybrid_smooth = nn.Conv2d(2, 2, 7, padding=3)
        # self.smooth = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.smooth = ConvBN(dim, dim, kernel_size=5)
        self.proj = SeparableConvBN(dim, dim, kernel_size=7)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps, 0, 0), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)

        # high frequency
        q_h = self.local1(x)
        k_h = self.local2(x)
        local = k_h + q_h
        high_freq = local[:, :, :H, :W]

        # low frequency
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q_h = rearrange(q_h, 'b ( h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                        h=self.num_heads,
                        d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws,
                        ws2=self.ws)  # 64,32,64,16
        k_h = rearrange(k_h, 'b ( h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                        h=self.num_heads,
                        d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws,
                        ws2=self.ws)  # 64,32,64,16
        # print("qkv shape is :%d\n",qkv.shape)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws,
                            ws2=self.ws)  # 64,32,64,16
        dots_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        dots_d = (q @ k.transpose(-2, -1)) * self.scale
        dots = dots_d + dots_h

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        low_freq = attn[:, :, :H, :W]

        # adaptive frequency fusion
        feat = low_freq + high_freq
        # 1.attention cat
        avg_attn = torch.mean(feat, dim=1, keepdim=True)
        max_attn, _ = torch.max(feat, dim=1, keepdim=True)
        attn = torch.cat([avg_attn, max_attn], dim=1)
        # 2.attention hybrid
        sig = self.hybrid_smooth(attn).sigmoid()
        # sig = attn.sigmoid()
        # 3.attention select
        out = low_freq * sig[:, 0, :, :].unsqueeze(1) + high_freq * sig[:, 1, :, :].unsqueeze(1)
        out = self.smooth(out)
        out = self.proj(out)

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FeatureAdaptMixer(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelationAwareFusion(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, r=16):
        super(RelationAwareFusion, self).__init__()
        self.r = r
        self.pre_x = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.pre_res = ConvBN(in_channels, decode_channels, kernel_size=1)
        self.fc = nn.Linear(r * r, 2)
        self.eps = 1e-8
        self.smooth = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        # self.wight_path = "/home/wym/projects/unetformer-loss/fig_results"

    def forward(self, x, res):

        # 预处理：特征对齐和归一化
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pre_x(x)
        res = self.pre_res(res)
        # 自适应平均池化
        x_att = nn.AdaptiveAvgPool2d(1)(x)
        res_att = nn.AdaptiveAvgPool2d(1)(res)
        # 空间关系
        b, c, h, w = x_att.size()  # B,C,1,1
        space_x = torch.mean(x_att, dim=1).view(b, -1)
        space_res = torch.mean(res_att, dim=1).view(b, -1)
        space_rel = torch.cat([space_x, space_res], dim=1)
        space_rel = torch.sigmoid(space_rel)
        space_weights = space_rel
        # space_weights = torch.softmax(space_rel, dim=1)

        # 通道关系
        x_att_split = x_att.view(b, self.r, c // self.r)  # B,r,C/r
        res_att_split = res_att.view(b, self.r, c // self.r)  # B,r,C/r
        chl_affinity = torch.bmm(x_att_split, res_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        channel_rel = self.fc(chl_affinity)
        channel_rel = torch.sigmoid(channel_rel)
        channel_weights = channel_rel
        # channel_weights = torch.softmax(channel_rel, dim=1)

        # 归一化
        weights = torch.add(space_weights, channel_weights)
        fuse_weights = torch.softmax(weights, dim=1)
        # fuse_weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + self.eps)
        weighted_x = fuse_weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        weighted_res = fuse_weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * res
        out = weighted_x + weighted_res
        out = self.smooth(out)

        return out


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=256,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 num_heads=16):
        super(Decoder, self).__init__()

        self.decoder_channels = decode_channels
        # decoder
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        self.b4 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size)

        # self.p3 = WF(encoder_channels[-2], decode_channels)
        self.p3 = RelationAwareFusion(encoder_channels[-2], decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size)

        # self.p2 = WF(encoder_channels[-3], decode_channels)
        self.p2 = RelationAwareFusion(encoder_channels[-3], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        # self.p1 = WF(encoder_channels[-4], decode_channels)
        self.p1 = RelationAwareFusion(encoder_channels[-4], decode_channels)
        self.b1 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size)

        self.FRH = FeatureRefinementHead(decode_channels)
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w, num_class):

        if self.training:
            x = self.pre_conv(res4)

            deep_feature = x[:, -num_class:, :, :]
            x = self.b4(x)

            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)

            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            h2 = x
            x = self.p1(x, res1)
            x = self.b1(x)

            x = self.segmentation_head(x)

            # init
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah, deep_feature
        else:
            x = self.pre_conv(res4)
            x = self.b4(x)

            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)
            x = self.b1(x)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AdaptFormer(nn.Module):
    def __init__(self,
                 decode_channels=256,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 # backbone_name='tf_efficientnet_b7',
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 num_heads=8
                 ):
        super().__init__()

        # pretrained_cfg = timm.models.create_model(backbone_name).default_cfg
        # pretrained_cfg['file'] = '../../pretrain_weights/efficientnet.pth'
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes, num_heads)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah, edge = self.decoder(res1, res2, res3, res4, h, w, self.num_classes)
            return x, ah, edge
        else:
            x = self.decoder(res1, res2, res3, res4, h, w, self.num_classes)
            return x


if __name__ == "__main__":

    from thop import profile

    x = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = AdaptFormer(num_classes=2, decode_channels=256, num_heads=8)
    # print(net)
    out = net(x)
    print(out[0].shape, out[1].shape, out[2].shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
