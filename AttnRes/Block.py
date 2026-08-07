import torch
import torch.nn as nn

class Mod(nn.Module):
    def __init__(
            self,
            io_dim: int, 
            latent_dim: int, 
            num_layers: int, 
            normalize: bool, 
            dropout: float):
        super().__init__()
        self._ifacedim  = io_dim
        self._latent    = latent_dim
        self._nlayers   = num_layers
        self._normalize = normalize
        self._p         = dropout

        self._drop = nn.Dropout(self._p)

        self.inlayer = nn.Linear(self._ifacedim, self._latent, bias=False)
        if self._normalize:
            self.norm1 = nn.LayerNorm(self._latent)
        self.activation = nn.GELU()

        self.layers = nn.ModuleList(
            nn.Linear(self._latent, self._latent, bias=False) for _ in range(self._nlayers)
        )

        if self._normalize:
            self.norms = nn.ModuleList(
                nn.LayerNorm(self._latent) for _ in range(self._nlayers)
            )

        self.outlayer = nn.Linear(self._latent, self._ifacedim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inlayer(x)
        # Stupid implementation of this but it's a personal repo so who cares
        if self._normalize:
            x = self.norm1(x)
            x = self.activation(x)
            x = self._drop(x)
            for norm, layer in zip(self.norms, self.layers):
                x = layer(x)
                x = norm(x)
                x = self.activation(x)
                x = self._drop(x)

            x = self.outlayer(x)
            return x
        
        else:
            x = self.activation(x)
            x = self._drop(x)
            for norm, layer in zip(self.norms, self.layers):
                x = layer(x)
                x = self.activation(x)
                x = self._drop(x)

            x = self.outlayer(x)
            return x