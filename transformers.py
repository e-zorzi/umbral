import torch
from math import sqrt

_MAX_CONTEXT = 16
_N_FEATURES = 17


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads=3, kq_embedding_size=4):
        super().__init__()
        self.Q_proj = torch.nn.Linear(_N_FEATURES, kq_embedding_size)
        self.K_proj = torch.nn.Linear(_N_FEATURES, kq_embedding_size)
        self.V_proj = torch.nn.Linear(_N_FEATURES, _N_FEATURES)

    def __call__(self, x: torch.tensor, return_weights=False):
        # assert x.shape[0] == _N_FEATURES
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        logits = torch.matmul(Q, K.T) / sqrt(_N_FEATURES)

        attention_weights = torch.softmax(logits, axis=-1)
        if return_weights:
            return (torch.matmul(attention_weights, V), attention_weights)
        else:
            return torch.matmul(attention_weights, V)


class TransformerBlock(torch.nn.Module):
    def __init__(self, n_heads=3, kq_embedding_size=4):
        super().__init__()
        self.multiheadattention = MultiHeadAttention(
            n_heads=n_heads, kq_embedding_size=kq_embedding_size
        )
        self.norm1 = torch.nn.RMSNorm(_N_FEATURES)
        self.norm2 = torch.nn.RMSNorm(_N_FEATURES)
        self.mlp = [
            torch.nn.Linear(_N_FEATURES, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, _N_FEATURES),
            torch.nn.SiLU(),
        ]

    def __call__(self, x: torch.tensor):
        y = self.multiheadattention(x)
        z = self.norm1(x + y)
        projected = z
        for layer in self.mlp:
            projected = layer(projected)
        return self.norm2(projected + z)
