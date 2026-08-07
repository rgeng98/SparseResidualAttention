import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, token_dim, dropout, aggregation = "LAST"):
        super().__init__()
        self.k = nn.Linear(token_dim, token_dim, bias=False)
        self.q = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, token_dim, bias=False)

        self._p     = dropout
        self._agg   = aggregation

    def forward(self, x):
        # input should be x: (B, S, H)
        k, q, v = self.k(x), self.q(x), self.v(x)

        attention_connection = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self._p if self.training else 0.0)

        if self._agg.upper() == "MEAN":
            return attention_connection.mean(dim = -2)
        elif self._agg.upper() == "LAST":
            return attention_connection[:,-1,:]