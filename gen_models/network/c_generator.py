
class Generator(nn.Module):
    def __init__(self, base_channels=64, bottom_width=4, z_dim=120, shared_dim=128, n_classes=n_classes, channel=channel):
        super().__init__()
        n_chunks = 6
        self.z_chunk_size = z_dim // n_chunks
        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.bottom_width = bottom_width
        self.shared_emb = nn.Embedding(n_classes, shared_dim)
        self.proj_z = nn.Linear(self.z_chunk_size, 16 * base_channels * bottom_width ** 2)
        self.g_blocks = nn.ModuleList([
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 16 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 8 * base_channels),
            ]),

            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 8 * base_channels, 4 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 4 * base_channels, 2 * base_channels),
                ChannelSpatialSELayer(2*base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 2 * base_channels, base_channels),
                ChannelSpatialSELayer(base_channels),
            ]),
        ])
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, channel, kernel_size=1, padding=0)),
            nn.Tanh(),
        )

    def forward(self, z, y):
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        ys = [torch.cat([y, z], dim=1) for z in zs[1:]]
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block[0](h, ys[idx])
        h = self.proj_o(h)
        return h
