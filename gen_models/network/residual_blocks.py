class GResidualBlock(nn.Module):
    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.upsample_fn = nn.Upsample(scale_factor=2)     # upsample occurs in every gblock
        self.mixin = (in_channels != out_channels)
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
    def forward(self, x, y):
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)
        return h + x


class ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels,  preactivation=False,  downsample=None, bn=None, attention=None):
    super(ResidualConv, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    self.preactivation = preactivation
    self.activation = nn.LeakyReLU(0.2)
    self.downsample = downsample
    self.bn = bn
    self.attention=attention
    if self.bn==True:
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    elif self.bn==False:
        self.bn3 =  nn.InstanceNorm2d(out_channels)
        self.bn4 =  nn.InstanceNorm2d(out_channels)
    else:
        self.bn5=None
    if self.attention == True:
        self.attention1 = ChannelSpatialSELayer(out_channels)
    else:
        self.attention3 = None
    self.downsample_fn = nn.AvgPool2d(2,2)
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True))
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1,bias=True))
    self.learnable_sc = True if (in_channels != out_channels) or self.downsample else False
    if self.learnable_sc:
        self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
  def shortcut(self, x):
    if self.preactivation:
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.downsample:
            x = self.downsample_fn(x)
    else:
        if self.downsample:
            x = self.downsample_fn(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
    return x
  def forward(self, x):
    if self.bn ==True:
        x = self.bn1(x)
    elif self.bn ==False:
        x = self.bn3(x)
    else:
        x = x
    if self.preactivation:
        h = F.relu(x)
    else:
        h = x
    h = self.conv1(h)
    if self.bn ==True:
        h = self.bn2(h)
    elif self.bn==False:
        h = self.bn4(h)
    else:
        h = h
    h = self.conv2(self.activation(h))
    if self.downsample:
        h = self.downsample_fn(h)

    h = h + self.shortcut (x)
    return h
