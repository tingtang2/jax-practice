from flax import linen as nn
from jax import Array
from jax import numpy as jnp


class MLP(nn.Module):

    @nn.compact
    def __call__(self, x: Array) -> Array:
        input_size = x.shape[-1]
        n_hidden = 256
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=input_size)(x)
        return x


class MLPwTime(nn.Module):

    @nn.compact
    def __call__(self, x: Array, t: Array) -> Array:
        input_size = x.shape[-1]
        n_hidden = 256

        t = jnp.concatenate([
            t - 0.5,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)
        ],
                            axis=-1)

        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=input_size)(x)
        return x
