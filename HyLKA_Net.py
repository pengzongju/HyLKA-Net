import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class s_mixer(nn.Module):
  def __init__(self, num_channels,num_blocks):
    super(s_mixer, self).__init__()
    self.num_blocks = num_blocks
    self.dgce = DGCE(320)
    self.conv0 = nn.Conv2d(320,64,kernel_size=1)
    layer = []
    for i in range(self.num_blocks):
        layer.append(MAB(i * num_channels + num_channels,num_channels))

    self.net = nn.Sequential(*layer)


  def forward(self, X):
      shortcut = X.clone()
      for blk in self.net:
          Y = blk(X)
          X = torch.cat((X, Y), dim=1)

      X = self.conv0(X)
      return X+shortcut

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SGAB(nn.Module):
    def __init__(self, num_channels, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super(SGAB,self).__init__()
        i_feats = num_channels * 2

        self.Conv1 = nn.Conv2d(num_channels, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(num_channels, num_channels, 7, 1, 7 // 2, groups=num_channels)
        self.Conv2 = nn.Conv2d(num_channels, num_channels, 1, 1, 0)

        self.norm = LayerNorm(num_channels, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, num_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class GroupGLKA(nn.Module):
    def __init__(self, n_feats, num_channels,k=2, squeeze_factor=15):
        super(GroupGLKA,self).__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA9 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 9, 1, 9 // 2, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 11, stride=1, padding=(11 // 2) * 5, groups=n_feats // 4, dilation=5),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 4, dilation=4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 4, dilation=3),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 4, dilation=2),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4)
        self.X5 = nn.Conv2d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4)
        self.X7 = nn.Conv2d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4)
        self.X9 = nn.Conv2d(n_feats // 4, n_feats // 4, 9, 1, 9 // 2, groups=n_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))
        self.proj_last1 = nn.Sequential(
            nn.Conv2d(n_feats, num_channels, 1, 1, 0),nn.LeakyReLU(0.1))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3,a_4 = torch.chunk(a, 4, dim=1)

        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),self.LKA9(a_4) * self.X9(a_4)],
                      dim=1)


        x = self.proj_last(x * a) * self.scale + shortcut
        x= self.proj_last1(x)

        return x

    # MAB


class MAB(nn.Module):
    def __init__(
            self, n_feats,num_channels):
        super(MAB,self).__init__()

        self.LKA = GroupGLKA(n_feats,num_channels)

        self.LFE = SGAB(num_channels)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x

class DGCE(nn.Module):
    def __init__(self,channels):
        super(DGCE,self).__init__()
        self.first_conv = nn.Conv2d(channels,channels,kernel_size=1)
        self.dwconv_a = nn.Conv2d(channels//2,channels//2,kernel_size=3,padding=1,groups=channels//2)
        self.dwconv_b = nn.Conv2d(channels//2,channels//2,kernel_size=3,padding=1,groups=channels//2)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.last_conv = nn.Conv2d(channels,channels,kernel_size=1)

    def forward(self,x):
        out = self.first_conv(x)
        x1,x2=torch.chunk(out,2,dim=1)
        x1_processed = self.gelu(self.dwconv_a(x1))
        x2_processed = self.gelu(self.dwconv_b(x2))

        x1_enhanced = x1*x2_processed
        x2_enhanced = x2*x1_processed

        out1 = torch.cat([x1_enhanced,x2_enhanced],dim=1)

        out = self.relu(self.last_conv(out1))

        return out+x

class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim, MHSA_params['num_heads'], MHSA_params['dropout'], bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    def SAI2Token(self, buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes ** 2, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer

class AltFilter1(nn.Module):
    def __init__(self, angRes, channels,MHSA_params):
        super(AltFilter1, self).__init__()
        self.angRes = angRes
        self.channels =channels
        self.dgce = DGCE(channels)
        self.S_Mixer = s_mixer(64,4)
        self.A_Mixer =AngTrans(channels, angRes, MHSA_params)
        self.E_Mixer = MAB(64,64)
        #self.SFB = SFC_B(4*channels)
        self.conv_fuse = nn.Conv3d(2*channels,channels,kernel_size=1,stride=1)

    def forward(self,buffer):
        shortcut = buffer
        [b,c,n,h,w] = buffer.size()

        # 角度特征提取

        buffer_A = self.A_Mixer(buffer)
        buffer_A = self.A_Mixer(buffer_A)
        buffer1 = rearrange(buffer_A, 'b c (u v) h w -> (b h w) c u v ', u=self.angRes, v=self.angRes, b=b, h=h, w=w)
        buffer_A1 = rearrange(buffer1, '(b h w) c u v -> b c (u v) h w', u=self.angRes, v=self.angRes, b=b, h=h, w=w)


        #空间特征提取
        buffer_S = rearrange(buffer_A1,'b c (u v) h w -> (b u v) c h w',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        # buffer_S = self.liner_in(buffer_S)
        buffer_S2 = self.S_Mixer(buffer_S)
        # buffer_S = self.liner_out(buffer_S)
        buffer_S2 = rearrange(buffer_S2,'(b u v) c h w -> b c (u v) h w',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        #水平EPI特征提取
        buffer_H_E = rearrange(buffer,'b c (u v) h w -> (b v w) c u h',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        buffer_H_E2 = self.E_Mixer(buffer_H_E)
        buffer_H_E2 = self.E_Mixer(buffer_H_E2)
        buffer_H_E2 = rearrange(buffer_H_E2,'(b v w) c u h -> b c (u v) h w',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        #垂直EPI特征提取
        buffer_V_E = rearrange(buffer_H_E2,'b c (u v) h w -> (b u h) c v w',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        buffer_V_E2 = self.E_Mixer(buffer_V_E)
        buffer_V_E2 = self.E_Mixer(buffer_V_E2)
        buffer_V_E2 = rearrange(buffer_V_E2 , '(b u h) c v w -> b c (u v) h w', u=self.angRes, v=self.angRes,b=b,h=h,w=w)

        out = torch.cat((buffer_S2, buffer_V_E2),dim=1)
        out = self.conv_fuse(out)+shortcut
        return out
# if __name__ == '__main__':
#     x = torch.randn(1,64,25,32,32)
#     model = AltFilter(5,64)
#     print(model(x).shape)

#低频残差
def LF_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale

class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)

class get_model(nn.Module):
    def __init__(self,args): #args
        super(get_model, self).__init__()
        channels = 64
        self.channels = channels
        #self.angRes=angRes
        #self.scale=scale
        self.angRes = args.angRes_in
        self.scale = args.scale_factor
        ang_num_heads=4
        ang_mlp_ratio=4
        ang_sr_ratio=1
        attn_drop_rate=0.0
        drop_rate=0.0
        drop_path_rate=0.1
        patch_size=32
        self.ang_num_heads = ang_num_heads
        self.ang_mlp_ratio = ang_mlp_ratio
        self.ang_sr_ratio = ang_sr_ratio
        self.attn_drop_rate = attn_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size

        self.pos_encoding = PositionEncoding(temperature=10000)

        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.


        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Bl0ck1 = AltFilter1(self.angRes, self.channels, self.MHSA_params)
        self.Bl0ck2 = AltFilter1(self.angRes, self.channels, self.MHSA_params)
        self.Bl0ck3 = AltFilter1(self.angRes, self.channels, self.MHSA_params)
        self.Bl0ck4 = AltFilter1(self.angRes, self.channels, self.MHSA_params)
        self.Bl0ck5 = AltFilter1(self.angRes, self.channels, self.MHSA_params)

        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )
    def forward(self, lr,info=None): #info=None
        lr = rearrange(lr, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = lr.size()
        sr_y = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')
        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)
        sr_y = rearrange(sr_y, 'b c u v h w -> b c (u h) (v w)', u=u, v=v)
        # Initial Feature Extraction
        x = rearrange(lr, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        ang_position = self.pos_encoding(buffer, dim=[2], token_dim=self.channels)
        for m in self.modules():
            m.ang_position = ang_position


        #Deep Feature Extraction
        out1 = self.Bl0ck1(buffer)
        out2 = self.Bl0ck2(out1)
        out3 = self.Bl0ck3(out2)
        out4 = self.Bl0ck4(out3)
        out5 = self.Bl0ck5(out4)
        # buffer_out = rearrange(out5,'b c (u v) h w -> (b u v) c h w',u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        # UP-Sampling
        out = rearrange(out5, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes,v=self.angRes,b=b,h=h,w=w)
        y = self.upsampling(out) + sr_y
        return y

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, degrade_info=None):
        loss = self.criterion_Loss(SR, HR)
        return loss
def weights_init(m):
    pass


'''
from thop import profile
model = get_model(5,4)
input = torch.randn(1,1,160,160)
flops, params = profile(model, inputs=(input,))
print(flops/1e9,params/1e6) #flops单位G，para单位M
'''




























