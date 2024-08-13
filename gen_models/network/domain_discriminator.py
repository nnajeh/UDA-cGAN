from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, channel= 3, filters=[64, 128, 256, 512, 1024],bn=None,  attention=None, dim=64):
      
        super().__init__()
      
        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes, 16 * base_channels*4*4))
      
        self.d_blocks = nn.Sequential(
            ResidualConv(channel, filters[0], preactivation=False, downsample=True,bn=False, attention=attention),
            ResidualConv(filters[0], filters[1], preactivation=True, downsample=True,bn=False, attention=attention),
            ResidualConv(filters[1], filters[2], preactivation=True, downsample=True,bn=False, attention=attention),
            ResidualConv(filters[2], filters[3], preactivation=True, downsample=True,bn=False, attention=attention),
            ChannelSpatialSELayer(filters[3]),
            ResidualConv(filters[3], filters[4], preactivation=True, downsample=True,bn=False, attention=attention),
            ChannelSpatialSELayer(filters[4]),
            ResidualConv(filters[4], filters[4], preactivation=True, downsample=False,bn=False, attention=attention),
            ChannelSpatialSELayer(filters[4]),
            nn.ReLU(inplace=True))
      
        self.proj_o = nn.utils.spectral_norm(nn.Linear(16 * base_channels*4*4, 1))
        self.fc1 = nn.Sequential(nn.Linear(base_channels *16*4*4 , 64),
                                 nn.BatchNorm1d(64),
                                 nn.Linear(64, 1),
                                 nn.ReLU(True))
      
    def extract_features(self, x):
        h = x.view(-1, channel, 128, 128)
        h = self.d_blocks(h)
        #h = torch.sum(h, dim=[2, 3])
        h = h.view(-1, 4*4*16*base_channels)#
        return h

  
    def forward(self, x, y=None, alpha=None):
       h = self.extract_features(x)
       uncond_out = self.proj_o(h)
       if y is None:
         return uncond_out

       cond_out = torch.sum(self.shared_emb(y.to(torch.int64)) * h, dim=1, keepdim=True)
       out = uncond_out + cond_out
       out = out.view(-1)

       if alpha is None:
            return out

       reverse_features = ReverseLayerF.apply(h, alpha)
       domain_prediction = self.fc1(reverse_features)
       return out, domain_prediction
