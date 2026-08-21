import torch
import torch.nn as nn
import AttnRes.Attention as A
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, latent_dim: int, drop: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, 4*latent_dim, bias=False)
        self.out = nn.Linear(latent_dim*4, latent_dim, bias=False)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.out(x)
        return self.drop(x)

# Next task is plugging this in and troubleshooting. KQ Norms should help
class MHA(nn.Module):
    def __init__(self, embedding_dim: int, n_heads:int):
        super().__init__()
        
        self.d_head = embedding_dim // n_heads
        self.n_heads = n_heads
        self.qkv  = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.qnorm = nn.RMSNorm(self.d_head)
        self.knorm = nn.RMSNorm(self.d_head)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1,2)

        q = self.qnorm(q)
        k = self.knorm(k)

        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).reshape(B, T, C)
        return self.proj(out)
# Decoder transformer as defined in Attention is All You Need
class Transformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            n_heads: int,
            dropout: float = 0.1,
            n_layers: int | None = None
        ):
        super().__init__()

        assert embedding_dim % n_heads == 0
        self.attn = torch.nn.MultiheadAttention(
            embedding_dim,
            n_heads,
            dropout=dropout,
            bias=False,
            batch_first=True
        )

        self.mlp = MLP(
            embedding_dim,
            dropout
        )

        self.norm1 = torch.nn.LayerNorm( embedding_dim )
        self.norm2 = torch.nn.LayerNorm( embedding_dim )

        # GPT-2 style residual scaling: the weights that write straight back
        # into the residual stream (attn's out_proj, mlp's final linear) get
        # shrunk by 1/sqrt(2*n_layers) at init, so that summing n_layers of
        # these contributions doesn't make the residual stream's variance
        # grow with depth.
        if n_layers is not None:
            scale = (2 * n_layers) ** 0.5
            with torch.no_grad():
                self.attn.out_proj.weight /= scale
                self.mlp.out.weight /= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        L = x.size(1)  # sequence length (dim 0 if batch_first=False)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device, dtype=x.dtype
        )
        
        y, _ = self.attn(x, x, x, is_causal=True, attn_mask=causal_mask)
        x = self.norm1(x + y)

        return self.norm2( self.mlp( x ) + x)
    
    

class LanguageModel(nn.Module):
    def __init__(
            self,
            num_embeddings: int, embedding_dim: int, n_heads: int,
            max_seq_len: int,
            attn_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        # Embedding module. 
        self.stem = nn.Embedding(
            num_embeddings,
            embedding_dim
        )
        # GPT-2-style small-std init. nn.Embedding's default init is N(0, 1),
        # which is fine for an embedding table on its own but far too large
        # once this weight also serves as the unembedding matrix
        nn.init.normal_(self.stem.weight, mean=0.0, std=0.02)

        # Learned positional embedding, since the attention layers themselves
        # (causal mask aside) carry no notion of token order.
        self.pos_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )

        assert embedding_dim % n_heads == 0

        self.Transformers = nn.ModuleList(
            Transformer(
                embedding_dim,
                n_heads,
                dropout=dropout,
                n_layers=attn_layers
            ) for _ in range(attn_layers)
        )

        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(
                embedding_dim
            ) for _ in range(attn_layers)
        )
        
        self.attn_residual = A.AttentionModule(embedding_dim, dropout, n_layers=attn_layers)
        
        self.activation = nn.GELU()
        self.output = nn.Linear(embedding_dim, num_embeddings, bias=False)

        self.output.weight = self.stem.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) token ids
        _, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.stem(x) + self.pos_embedding(positions).unsqueeze(0)
        residual_list = torch.tensor([], device=x.device)
        for transformer, norm in zip(self.Transformers, self.layer_norms):
            # Starting with the architecture that we know should work well.
            y = transformer(x)

            resid = x.reshape(-1, x.shape[-1])

            residual_list = torch.cat( ( residual_list, resid.unsqueeze(1) ), dim = 1 ) # Shape (B*S, R, H)

            residual_value = self.attn_residual(residual_list).reshape(x.shape)
            x = y + residual_value

        # print(x.shape)
        return self.output(x)

