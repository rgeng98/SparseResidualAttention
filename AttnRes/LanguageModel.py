import torch
import torch.nn as nn
import AttnRes.Module as M
import matplotlib.pyplot as plt

class LanguageModelSparseAttentionResNet(nn.Module):
    def __init__(
            self,
            num_embeddings: int, embedding_dim: int,
            resblocks: int, resdim: int, reslayers: int, latentdim: int,
            normalize: bool = True, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=1
        )
        self.backbone = M.SparseAttentionResNet(
            indim=embedding_dim,
            outdim=num_embeddings,
            resblocks=resblocks,
            resdim=resdim,
            reslayers=reslayers,
            latentdim=latentdim,
            normalize=normalize,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        x = self.stem(x)
        return self.backbone(x)
