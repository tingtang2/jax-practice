from functools import partial
from typing import Dict

import jax
import optax
from flax import linen as nn
from jax import Array, jit
from jax import numpy as jnp
from jax import random, value_and_grad
from plotting_utils import plot_heatmap

from model import MLP


def loss_fn(params: Dict, model: nn.Module, rng: Array, data: Array):
    # noise data
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    noised_data = 0.01 * data + noise

    # predict noise level
    output = model.apply(params, noised_data)
    return jnp.mean((noise - output)**2)


def sample(rng: Array, n_samples: int, model: nn.Module, params: Dict):
    rng, step_rng = random.split(rng)
    noised_data = random.normal(step_rng, (n_samples, 2))
    predicted_noise = model.apply(params, noised_data)

    data = 100 * (noised_data - predicted_noise)
    return data


def run_experiment(seed: int, dataset: Array):

    @partial(jit, static_argnums=[4])
    def update_weights(
        params: Dict,
        rng: Array,
        batch: Array,
        opt_state: optax.OptState,
        model: nn.Module,
    ):
        val, grads = value_and_grad(loss_fn)(params, model, rng, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return val, params, opt_state

    # init model and optimizer
    rng = random.PRNGKey(seed)
    rng, model_rng = random.split(rng)

    denoiser = MLP()
    x = jnp.empty((10, 2))
    params = denoiser.init(model_rng, x)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params=params)

    # run training loop
    n_epochs = 60000
    losses = []

    for k in range(n_epochs):
        rng, step_rng = random.split(rng)
        loss, params, opt_state = update_weights(params, step_rng, dataset,
                                                 opt_state, denoiser)
        losses.append(loss)

        if (k + 1) % 5000 == 0:
            mean_loss = jnp.mean(jnp.array(losses))
            losses = []
            print(f'Epoch {k+1}, Loss: {mean_loss:0.5f}')

    # sample from model
    n_samples = 1000
    rng, s_rng = random.split(rng)
    samples = sample(s_rng, n_samples=n_samples, model=denoiser, params=params)
    plot_heatmap(samples)
