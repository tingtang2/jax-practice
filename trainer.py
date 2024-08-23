from functools import partial
from typing import Dict

import jax
import optax
from flax import linen as nn
from jax import Array, jit
from jax import numpy as jnp
import numpy as np
from jax import random, value_and_grad
from plotting_utils import plot_heatmap, heatmap_data

from matplotlib import pyplot as plt

from model import MLP, MLPwTime
import matplotlib.animation as animation


def better_sample_with_time(rng: Array, n_samples: int, model: nn.Module,
                            params: Dict, alphas: Array, betas: Array):
    rng, step_rng = random.split(rng)
    all_outputs = np.zeros((len(betas) + 1, n_samples, 2))
    noised_data = random.normal(step_rng, (n_samples, 2))
    all_outputs[0, :, :] = noised_data

    for i in range(len(betas)):
        beta = betas[-i]
        alpha = alphas[-i] * jnp.ones((noised_data.shape[0], 1))
        noise_guess = model.apply(params, noised_data, alpha)
        rng, step_rng = random.split(rng)
        new_noise = random.normal(step_rng, noised_data.shape)
        noised_data = 1 / (1 - beta)**0.5 * (noised_data - beta /
                                             (1 - alpha)**0.5 * noise_guess)

        if i < len(betas) - 1:
            noised_data += beta**0.5 * new_noise

        all_outputs[i + 1, :, :] = noised_data

    return noised_data, all_outputs


def sample_with_time(rng: Array, n_samples: int, model: nn.Module,
                     params: Dict, alphas: Array):
    rng, step_rng = random.split(rng)
    noised_data = random.normal(step_rng, (n_samples, 2))

    for i in range(len(alphas)):
        alpha = alphas[-i] * jnp.ones((noised_data.shape[0], 1))
        noise_guess = model.apply(params, noised_data, alpha)
        denoised_guess = 1 / alpha**0.5 * (noised_data - noise_guess *
                                           (1 - alpha)**0.5)
        rng, step_rng = random.split(rng)

        if i < len(alphas) - 1:
            new_noise = random.normal(step_rng, noised_data.shape)
            next_alpha = alphas[-i - 1]
            new_noised = denoised_guess * next_alpha**0.5 + (
                1 - alpha)**0.5 * new_noise
            noised_data = new_noised

    return noised_data


def better_loss_fn(params: Dict, model: nn.Module, rng: Array, data: Array,
                   alphas: Array):
    rng, step_rng = random.split(rng)
    alpha = random.choice(key=step_rng, a=alphas, shape=(data.shape[0], 1))

    rng, step_rng = random.split(rng)
    noise = random.normal(key=step_rng, shape=data.shape)

    noised_data = data * alpha**0.5 + noise * (1 - alpha)**0.5

    output = model.apply(params, noised_data, alpha)

    return jnp.mean((noise - output)**2)


def get_alpha_beta_schedule(n: int,
                            beta_min: float = 0.1,
                            beta_max: float = 20.0):
    # We interpolate between taking small noising steps first (of size beta_min/N)
    # and taking larger steps at the end (of magnitude beta_max/N)
    # one can use any kind of beta scheduling here, and finding the best one is an open research question
    # The one we take is inspired by the scheduling chosen inhttps://github.com/yang-song/score_sde and
    # has proven itself in practice

    betas = jnp.array([
        beta_min / n + i / (n * (n - 1)) * (beta_max - beta_min)
        for i in range(n)
    ])
    # Note that N should be at least of the size beta_max so that all betas are positive

    alphas = jnp.cumprod(1 - betas)
    return alphas, betas


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


def run_bad_experiment(seed: int, dataset: Array):

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
            print(f'Epoch {k+1:>5}, Loss: {mean_loss:0.5f}')

    # sample from model
    n_samples = 1000
    rng, s_rng = random.split(rng)
    samples = sample(s_rng, n_samples=n_samples, model=denoiser, params=params)
    plot_heatmap(samples)


def run_experiment(seed: int, dataset: Array):

    @partial(jit, static_argnums=[4])
    def update_weights(params: Dict, rng: Array, batch: Array,
                       opt_state: optax.OptState, model: nn.Module,
                       alphas: Array):
        val, grads = value_and_grad(better_loss_fn)(params, model, rng, batch,
                                                    alphas)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return val, params, opt_state

    # init model and optimizer
    rng = random.PRNGKey(seed)
    rng, model_rng = random.split(rng)

    denoiser = MLPwTime()
    x = jnp.empty((10, 2))
    t = jnp.empty((10, 1))
    params = denoiser.init(model_rng, x, t)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params=params)

    alphas, betas = get_alpha_beta_schedule(n=100)
    plt.plot(alphas, label="Amount Signal")
    plt.plot(1 - alphas, label="Amount Noise")
    plt.legend()
    plt.savefig('alpha_beta_schedule.png')
    plt.clf()

    # run training loop
    n_epochs = 60000
    losses = []

    for k in range(n_epochs):
        rng, step_rng = random.split(rng)
        loss, params, opt_state = update_weights(params, step_rng, dataset,
                                                 opt_state, denoiser, alphas)
        losses.append(loss)

        if (k + 1) % 5000 == 0:
            mean_loss = jnp.mean(jnp.array(losses))
            losses = []
            print(f'Epoch {k+1:>5}, Loss: {mean_loss:0.5f}')

    # sample from model
    n_samples = 1000
    rng, s_rng = random.split(rng)
    samples, all_outputs = better_sample_with_time(s_rng,
                                                   n_samples=n_samples,
                                                   model=denoiser,
                                                   params=params,
                                                   alphas=alphas,
                                                   betas=betas)
    # save gif
    fig = plt.figure(figsize=(8, 8))
    im = plot_heatmap(samples)

    def animate(frame):
        im.set_array(heatmap_data(all_outputs[frame, :, :]))
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=all_outputs.shape[0])
    anim.save('samples.gif', fps=10)
    # anim.save('samples.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
