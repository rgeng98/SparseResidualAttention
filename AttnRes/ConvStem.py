import torch
import torch.nn as nn
import AttnRes.Module as M
from AttnRes.GPT import Transformer
import AttnRes.Attention as Attn

class ConvStem(nn.Module):
    def __init__(self, in_h, in_w = None, in_channels=1, channels=(16, 32, 64, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, c), num_channels=c))
            layers.append(nn.GELU())
            layers.append(nn.Dropout2d(dropout))
            prev_c = c
        
        if in_w is not None:
            layers.append(nn.AdaptiveAvgPool2d(1))

            self.conv = nn.Sequential(*layers)
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, in_h, in_w)
                self.out_dim = self.conv(dummy).flatten(start_dim=1).shape[1]
        else:
            with torch.no_grad():
                self.conv = nn.Sequential(*layers)
                dummy = torch.zeros(1, in_channels, in_h, in_h)
                print(dummy.shape)
                self.out_dim = self.conv(dummy).shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv(x)
        return x


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

        x = torch.flatten(x, start_dim=1)

        return self.backbone(x)


class LanguageModel(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            conv_channels=(1, 1), # Extend the conv channels tuple to decrease the computation size of the transformers
            n_transformers = 8,
            n_heads = 2,
            normalize=True, dropout=0.1):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=1
        )

        # Generalize connections between successive tokens before passing through transformers. This enables the stride to reduce the computational expense
        # and memory footprint of these transformers
        self.stem = ConvStem(in_h=embedding_dim, in_channels=1, channels=conv_channels, dropout=dropout)
        print(self.stem.out_dim)
        # Transformer backbone responsible for uncovering the complex relations between the convolutional stem's outputs
        self.backbone = nn.ModuleList(
            Transformer(
                self.stem.out_dim,
                n_heads,
                dropout=dropout
            ) for _ in range(n_transformers)
        )

        self._residual_attention = Attn.AttentionModule(
            self.stem.out_dim,
            dropout,
            "LAST"
        )
        
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(
                self.stem.out_dim
            ) for _ in range(n_transformers)
        )

        self.out = nn.Linear(self.stem.out_dim, num_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, 1) -> (B, S, T)
        x = self.embed(x)

        # x: (B, S, T) -> (B, 1, S, T)
        x = x.unsqueeze(1)

        # Convolutional stem to compress data
        x = self.stem(x)

        # (B, 1, S, T)
        residual_tracker = x

        # (B, S, T)
        x = x.squeeze(1)

        for i, mod in enumerate(self.backbone):
            x = mod(x)
            
            # Perform the attention mechanism on the sequence of residual tokens and complete residual connection
            if i > 0:
                x = self.layer_norms[i]( x + self._residual_attention( residual_tracker ) )
            else:
                x = self.layer_norms[i]( x + residual_tracker[:,-1,:] )

            residual_tracker = torch.cat( (residual_tracker, x.unsqueeze(1)), dim = 1 )

        return self.out( x[:,-1,:] )

