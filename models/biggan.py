import os
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from .batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=F.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.pool = nn.MaxPool2d(2, 2)
        self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False))
        self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        N = height * width

        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = phi.view(m_batchsize, -1, N // 4)
        theta = theta.view(m_batchsize, -1, N)
        theta = theta.permute(0, 2, 1)
        attention = self.softmax(torch.bmm(theta, phi))  # BX (N) X (N)
        g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
        attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize, -1, width, height)
        out = self.o_conv(attn_g)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = BatchNorm2d(num_features, affine=False, eps=1e-4)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        bn=True,
        activation=F.relu,
        upsample=True,
        downsample=False,
        z_dim=148,
    ):
        super().__init__()

        self.conv0 = SpectralNorm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
        )
        self.conv1 = SpectralNorm(
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
        )

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
            self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        # return out
        if self.upsample:
            # TODO different form papers
            out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip


class Generator(nn.Module):
    def __init__(self, code_dim, pixels, n_class=1000, chn=96, debug=False):
        super().__init__()

        self.linear = nn.Linear(n_class, 128, bias=False)

        if debug:
            chn = 8

        self.first_view = 16 * chn

        z_dim = code_dim + {128: 28, 256: 8, 512: 16}.get(pixels)

        self.G_linear = SpectralNorm(nn.Linear(20 if pixels != 512 else 16, 4 * 4 * 16 * chn))

        module_list = [
            GBlock(16 * chn, 16 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(16 * chn, 8 * chn, n_class=n_class, z_dim=z_dim),
        ]
        if pixels != 128:
            module_list.append(GBlock(8 * chn, 8 * chn, n_class=n_class, z_dim=z_dim))
        
        module_list += [
            GBlock(8 * chn, 4 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim),
        ]
        if pixels == 512:
            module_list.append(GBlock(1 * chn, 1 * chn, n_class=n_class, z_dim=z_dim))

        self.GBlock = nn.ModuleList(module_list)

        sa_scale = 2 if pixels != 512 else 4

        self.sa_id = 5 if pixels == 256 else 4
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(sa_scale * chn)
        self.ScaledCrossReplicaBN = BatchNorm2d(1 * chn, eps=1e-4)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.chunk(input, self.num_split, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)
