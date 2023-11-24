from typing import Tuple


class Codebook(nn.Module):
    """Codebook."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dimension: int,
    ) -> None:
        """Initializes the module."""

        super().__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dimension = embedding_dimension

        self.embedding = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dimension,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')

        tokens = torch.mean(torch.square(self.embedding.weight - x.unsqueeze(1)), dim=2).argmin(dim=1)
        quantized = x + (self.embedding(tokens) - x).detach()
        quantized = quantized.view(B, C, H, W)
        tokens = tokens.view(B, H, W)
        
        return quantized, tokens


class ResidualBlock(nn.Module):
    """Residual blcok."""

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        """Initializes the module."""

        super().__init__()

        self.in_channels = in_channels

        self.convolution_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.convolution_2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        x = x + F.leaky_relu(self.convolution_1(x))
        x = x + F.leaky_relu(self.convolution_2(x))

        return x


class DownsamplingBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initializes the module."""

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=4,
                num_channels=out_channels,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return self.layers(x)


class UpsamplingBlock(nn.Module):
    """Upsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initializes the module."""

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=4, 
                num_channels=in_channels,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return self.layers(x)


class VQVAE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        codebook_size: int,
    ) -> None:
        """Initializes the module."""

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.codebook_size = codebook_size

        self.encoder = nn.Sequential(
            DownsamplingBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
            ),
            DownsamplingBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
            ),
            ResidualBlock(in_channels=hidden_channels),
            ResidualBlock(in_channels=hidden_channels),
        )

        self.decoder = nn.Sequential(
            ResidualBlock(in_channels=hidden_channels),
            ResidualBlock(in_channels=hidden_channels),
            UpsamplingBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
            ),
            UpsamplingBlock(
                in_channels=hidden_channels,
                out_channels=in_channels,
            ),
        )

        self.codebook = Codebook(
            codebook_size=codebook_size,
            embedding_dimension=hidden_channels,
        )
    
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""

        embedding = self.encoder(x)
        quantized, tokens = self.codebook(embedding)
        reconstruction = self.decoder(quantized)

        return reconstruction, embedding, quantized, tokens
