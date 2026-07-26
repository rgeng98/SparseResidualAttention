import torch
import torch.nn as nn
import AttnRes.Block as B
import AttnRes.Attention as Attn

class SparseAttentionResNet(nn.Module):
    def __init__(self, indim, outdim, resblocks, resdim, reslayers, latentdim, normalize=True, dropout=0.1):
        super().__init__()
        self.inlayer = nn.Linear(indim, latentdim)
        self.norm1   = nn.LayerNorm(latentdim)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.GELU()
        # Create our module list of residual blocks
        self.residual_modules = nn.ModuleList(
            B.Mod(
                latentdim,
                resdim,
                reslayers,
                normalize,
                dropout
            ) for _ in range(resblocks)
        )
        self.outlayer = nn.Linear(latentdim, outdim)

        self._residual_attention = Attn.AttentionModule(
            latentdim,
            dropout,
            "LAST"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the input data into a latent dimension
        x = self.inlayer(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.drop(x)
        # Reshape to (B, 1, H) to concatenate along dim=1
        residual_tracker = x.unsqueeze(1)
        for i, mod in enumerate(self.residual_modules):
            x = mod(x)
            residual_tracker = torch.cat( (residual_tracker, x.unsqueeze(1)), dim = -2 )
            # Perform the attention mechanism on the sequence of residual tokens and complete residual connection
            if i > 0:
                x = x + self._residual_attention( residual_tracker )
            else:
                x = x + residual_tracker[:,-1,:]

        return self.outlayer(x)
    
    