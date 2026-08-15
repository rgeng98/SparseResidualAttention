import torch
import torch.nn as nn
import AttnRes.Attention as A
import matplotlib.pyplot as plt

# Decoder transformer as defined in Attention is All You Need
class Transformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int, n_heads: int,
            attn_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        assert embedding_dim % n_heads == 0
        self.attn_layers = nn.ModuleList(
            torch.nn.MultiheadAttention(
                embedding_dim,
                n_heads,
                dropout=dropout,
                bias=False,
                batch_first=True
            ) for _ in range(attn_layers)
        )

        self.linear = torch.nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.norm_layers = nn.ModuleList(
            torch.nn.LayerNorm(
                embedding_dim
            ) for _ in range(attn_layers)
        )

        self.norm = torch.nn.LayerNorm( embedding_dim ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        L = x.size(1)  # sequence length (dim 0 if batch_first=False)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device, dtype=x.dtype
        )
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            # print(x.shape)
            y, _ = attn(x, x, x, is_causal=True, attn_mask=causal_mask)
            x = norm(x + y)

        return self.norm( self.linear( x ) + x)
    
    

class LanguageModel(nn.Module):
    def __init__(
            self,
            num_embeddings: int, embedding_dim: int, n_heads: int,
            attn_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        # Embedding module
        self.stem = nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=1
        )

        assert embedding_dim % n_heads == 0

        self.Transformers = nn.ModuleList(
            Transformer(
                embedding_dim,
                n_heads,
                dropout=dropout
            ) for _ in range(attn_layers)
        )

        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(
                embedding_dim
            ) for _ in range(attn_layers)
        )
        
        self.attn_residual = A.AttentionModule(embedding_dim, dropout, aggregation="LAST")
        
        self.activation = nn.GELU()
        self.output = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        x = self.stem(x)
        residual_list = torch.tensor([], device=x.device)
        for transformer, norm in zip(self.Transformers, self.layer_norms):
            # Starting with the architecture that we know should work well.
            x = norm( transformer(x) + x )

            # resid = x.reshape(-1, x.shape[-1])

            # residual_list = torch.cat( ( residual_list, resid.unsqueeze(1) ), dim = 1 ) # Shape (B*S, R, H)

            # residual_value = self.attn_residual(residual_list).reshape(x.shape)
            # x = norm(y + residual_value)

        # print(x.shape)
        return self.output(x)

