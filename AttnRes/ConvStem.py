import torch
import torch.nn as nn
import AttnRes.Module as M


class ConvStem(nn.Module):
    def __init__(self, in_h, in_w, in_channels=1, channels=(16, 32, 64, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, c), num_channels=c))
            layers.append(nn.GELU())
            layers.append(nn.Dropout2d(dropout))
            prev_c = c
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_h, in_w)
            self.out_dim = self.conv(dummy).flatten(start_dim=1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, out_dim)
        x = self.conv(x)
        return torch.flatten(x, start_dim=1)


class ConvSparseAttentionResNet(nn.Module):
    def __init__(
            self,
            in_h, in_w,
            outdim, resblocks, resdim, reslayers, latentdim,
            in_channels=1, conv_channels=(16, 32, 64, 128),
            normalize=True, dropout=0.1):
        super().__init__()
        self.stem = ConvStem(in_h, in_w, in_channels, conv_channels, dropout)
        self.backbone = M.SparseAttentionResNet(
            indim=self.stem.out_dim,
            outdim=outdim,
            resblocks=resblocks,
            resdim=resdim,
            reslayers=reslayers,
            latentdim=latentdim,
            normalize=normalize,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.stem(x)
        return self.backbone(x)
