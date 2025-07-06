import jax
import jax.numpy as jnp
import equinox as eqx

_MAX_CONTEXT = 16
_N_FEATURES = 12


class MultiHeadAttention(eqx.Module):
    Q_stack: eqx.nn.Linear
    K_stack: eqx.nn.Linear
    V_stack: eqx.nn.Linear

    def __init__(self, key, n_heads=3, kq_embedding_size=4):
        key_Q, key_K, key_V = jax.random.split(key, 3)
        self.Q_stack = eqx.nn.Linear(_N_FEATURES, kq_embedding_size, key=key_Q)
        self.K_stack = eqx.nn.Linear(_N_FEATURES, kq_embedding_size, key=key_K)
        self.V_stack = eqx.nn.Linear(_N_FEATURES, _N_FEATURES, key=key_V)

    def __call__(self, x: jnp.array):
        assert x.shape[1] == _N_FEATURES and x.shape[0] == _MAX_CONTEXT
        Q = jax.vmap(self.Q_stack)(x)
        K = jax.vmap(self.K_stack)(x)
        V = jax.vmap(self.V_stack)(x)
        attention_weights = jax.nn.softmax(
            jnp.matmul(Q, K.T) / jnp.sqrt(_N_FEATURES), axis=1
        )
        attention_weights = attention_weights / (attention_weights.sum(axis=1))
        return jnp.matmul(attention_weights, V)


class TransformerBlock(eqx.Module):
    multiheadattention: MultiHeadAttention
    norm: eqx.nn.RMSNorm
    mlp: eqx.nn.Sequential

    def __init__(self, key, n_heads=3, kq_embedding_size=4):
        key, key_ma = jax.random.split(key, 2)
        key_linear1, key_linear2, key_linear3 = jax.random.split(key, 3)
        self.multiheadattention = MultiHeadAttention(
            key_ma, n_heads=n_heads, kq_embedding_size=kq_embedding_size
        )
        self.norm = eqx.nn.RMSNorm(_N_FEATURES)
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(_N_FEATURES, 128, key=key_linear1),
            eqx.nn.Linear(128, 128, key=key_linear2),
            eqx.nn.Linear(128, _N_FEATURES, key=key_linear3),
        ])

    def __call__(self, x: jnp.array):
        y = self.multiheadattention(x)
        z = jax.vmap(self.norm)(x + y)
        projected = jax.vmap(self.mlp)(z)
        return projected + z
