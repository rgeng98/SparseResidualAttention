import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, token_dim, dropout, n_layers: int | None = None):
        super().__init__()
        self.k = nn.Linear(token_dim, token_dim, bias=False)
        self.q = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, token_dim, bias=False)

        self._p     = dropout

        # Weight initialization
        if n_layers is not None:
            with torch.no_grad():
                self.v.weight /= (2 * n_layers) ** 0.5

    def forward(self, x):
        # input should be x: (B, S, H)
        k, q, v = self.k(x), self.q(x), self.v(x)

        attention_connection = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self._p if self.training else 0.0, attn_mask=None)

        return attention_connection[:,-1,:]