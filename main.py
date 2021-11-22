import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator=False,use_act=True,use_bn=True,**kwargs):
        super().__init__()
        self.use_bn = use_bn
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels,out_channels,**kwargs,bias= not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2,inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        )
    def forward(self,x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c*scale_factor**2,3,1,1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)
    def forward(self,x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = ConvBlock(in_c, in_c, kernel_size=3,stride=1,padding=1)
        self.block2 = ConvBlock(in_c, in_c, kernel_size=3,stride=1,padding=1,use_act=False)
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        return out+x

class Generator(nn.Module):
    def __init__(self, in_c=3,num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_c, num_channels, kernel_size=9, stride=1,padding=4,use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels,num_channels,kernel_size=3,stride=1,padding=1,use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2), UpsampleBlock(num_channels,scale_factor=2))
        self.final = ConvBlock(num_channels, in_c, kernel_size=9, stride=1,padding=4)
    def forward(self,x):
        initial = self.inital(x)
        x = self.residuals(initial)
        x = self.convblock(x)+initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))
class Discriminator(nn.Module):
    pass