import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 1. Continuous parameters (A is structurally a negative parameter matrix)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model)) # Skip connection vector

        # 2. Selective projections (Input-dependent B, C, and Delta)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False) 
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        # x shape: (B, S, d_model)
        b, s, d = x.shape
        
        # Initialize hidden state h: (B, d_model, d_state)
        h = torch.zeros(b, d, self.d_state, device=x.device)
        outputs = []

        A = -torch.exp(self.A_log) # Ensure stability (negative values)

        # Recurrent loop over sequence length (Inference mode style)
        for i in range(s):
            x_t = x[:, i, :] # Current token: (B, d_model)
            
            # Project x_t to get dynamic B, C, and Delta
            proj_out = self.x_proj(x_t)

            # Split into dynamic B (B, d_state), C (B, d_state), and raw Delta (B, d_model)
            B, C, delta_raw = torch.split(proj_out, [self.d_state, self.d_state, self.d_model], dim=-1)
            
            # Compute discretized Delta
            delta = F.softplus(self.dt_proj(delta_raw)) # (B, d_model)

            # Discretize A and B (Using simplified Euler or Zero-Order Hold approximation)
            # A shape: (d_model, d_state) -> exp(delta * A)
            dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0)) # (B, d_model, d_state)
            dB = delta.unsqueeze(-1) * B.unsqueeze(1)            # (B, d_model, d_state)

            # Update hidden state: h_t = dA * h_{t-1} + dB * x_t
            h = dA * h + dB * x_t.unsqueeze(-1)

            # Compute output token: y_t = C * h_t
            y_t = torch.einsum('bld,bd->bl', h, C) # (B, d_model)
            
            # Add skip connection (D)
            y_t = y_t + x_t * self.D
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1) # (B, S, d_model)

class LanguageModel(nn.Module):
    def __init__(
        self, 
        num_embeddings,
        embedding_dim,
        max_sequence_length,
        n_ssm = 2
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=1
        )
        self.positional_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
            max_norm=1
        )

        self.inTransformer = torch.nn.MultiheadAttention(
            embedding_dim,
            16,
            dropout=0.1,
            bias=False
        )
        self.ff = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.kobe = nn.ModuleList(
            [
                Mamba(embedding_dim) for _ in range(n_ssm)
            ]
        )

        self.norm = nn.ModuleList(
            [
                nn.LayerNorm(embedding_dim) for _ in range(n_ssm)
            ]
        )
        self.actfcn = nn.GELU()

        self.output_interp = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=x.device) # shape (t)
        tokens = self.embed(x) + self.positional_embedding(pos).unsqueeze(0) # (B, S, T)
        x, attn_output_weights = self.inTransformer(tokens, tokens, tokens)
        x = self.norm1( tokens + self.ff(x) )
        # Ridiculously simple stack of Mamba modules
        for mamba, ln  in zip(self.kobe, self.norm):
            # Sequential mamba modules 
            tokens = self.actfcn( ln( tokens + mamba(tokens) ) )
        
        return self.output_interp( tokens[:,-1,:] )


