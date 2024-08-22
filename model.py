from flax import linen as nn
from jax import Array


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
